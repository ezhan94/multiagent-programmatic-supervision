import torch
import torch.nn as nn

from models.utils import parse_model_params, get_params_str, cudafy_list, index_by_agent, get_macro_ohe
from models.utils import sample_gauss, nll_gauss, kld_gauss, sample_multinomial


class MACRO_SHARED_VRNN(nn.Module):
    """
    Our model that uses VRNN with programmatic weak supervision from labeling functions.
    
    In this version, the macro-intents for all agents are learned in one model (dec_macro).
    """

    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'm_dim', 'rnn_micro_dim', 'rnn_macro_dim', 'n_layers', 'n_agents']
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        x_dim = params['x_dim']
        y_dim = params['y_dim']
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        m_dim = params['m_dim']
        rnn_micro_dim = params['rnn_micro_dim']
        rnn_macro_dim = params['rnn_macro_dim']
        n_layers = params['n_layers']
        n_agents = params['n_agents']

        self.dec_macro = nn.Sequential(
            nn.Linear(y_dim+rnn_macro_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, m_dim),
            nn.LogSoftmax(dim=-1))

        self.enc = nn.ModuleList([nn.Sequential(
            nn.Linear(x_dim+y_dim+m_dim+rnn_micro_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.enc_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.enc_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.prior = nn.ModuleList([nn.Sequential(
            nn.Linear(y_dim+m_dim+rnn_micro_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.prior_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.prior_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(y_dim+m_dim+z_dim+rnn_micro_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_agents)])
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.gru_micro = nn.ModuleList([nn.GRU(x_dim+z_dim, rnn_micro_dim, n_layers) for i in range(n_agents)])
        self.gru_macro = nn.GRU(m_dim, rnn_macro_dim, n_layers)


    def forward(self, states, macro=None, hp=None):
        n_agents = self.params['n_agents']

        states_single = index_by_agent(states, n_agents)
        macro_shared = get_macro_ohe(macro, 1, self.params['m_dim']).squeeze()

        out = {}
        if hp['pretrain']:
            out['crossentropy_loss'] = 0
        else:
            out['kl_loss'] = 0
            out['recon_loss'] = 0
        
        h_micro = [torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_micro_dim']) for i in range(n_agents)]
        h_macro = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_macro_dim'])
        if self.params['cuda']:
            h_macro = h_macro.cuda()
            h_micro = cudafy_list(h_micro)

        for t in range(states.size(0)-1):
            x_t = states_single[t].clone()
            y_t = states[t].clone()
            m_t = macro_shared[t].clone()
            
            if hp['pretrain']:
                dec_macro_t = self.dec_macro(torch.cat([y_t, h_macro[-1]], 1))
                out['crossentropy_loss'] -= torch.sum(m_t*dec_macro_t)
    
                _, h_macro = self.gru_macro(torch.cat([m_t], 1).unsqueeze(0), h_macro)

            else:
                for i in range(n_agents):
                    enc_t = self.enc[i](torch.cat([x_t[i], y_t, m_t, h_micro[i][-1]], 1))
                    enc_mean_t = self.enc_mean[i](enc_t)
                    enc_std_t = self.enc_std[i](enc_t)

                    prior_t = self.prior[i](torch.cat([y_t, m_t, h_micro[i][-1]], 1))
                    prior_mean_t = self.prior_mean[i](prior_t)
                    prior_std_t = self.prior_std[i](prior_t)

                    z_t = sample_gauss(enc_mean_t, enc_std_t)

                    dec_t = self.dec[i](torch.cat([y_t, m_t, z_t, h_micro[i][-1]], 1))
                    dec_mean_t = self.dec_mean[i](dec_t)
                    dec_std_t = self.dec_std[i](dec_t)

                    _, h_micro[i] = self.gru_micro[i](torch.cat([x_t[i], z_t], 1).unsqueeze(0), h_micro[i])

                    out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                    out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t[i])

        return out


    def sample(self, states, macro, burn_in=0, fix_m=[]):
        n_agents = self.params['n_agents']

        macro_shared = get_macro_ohe(macro, 1, self.params['m_dim']).squeeze()

        if len(fix_m) == 0:
            fix_m = [-1]*n_agents

        h_micro = [torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_micro_dim']) for i in range(n_agents)]
        h_macro = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_macro_dim'])
        macro_intents = torch.zeros(macro.size())

        for t in range(states.size(0)-1):
            y_t = states[t].clone()
            m_t = macro_shared[t].clone()

            for i in range(n_agents):
                if t >= burn_in:
                    dec_macro_t = self.dec_macro(torch.cat([y_t, h_macro[-1]], 1))
                    m_t = sample_multinomial(torch.exp(dec_macro_t))

            macro_intents[t] = torch.max(m_t, -1)[1].unsqueeze(-1)
            _, h_macro = self.gru_macro(torch.cat([m_t], 1).unsqueeze(0), h_macro)

            for i in range(n_agents):
                prior_t = self.prior[i](torch.cat([y_t, m_t, h_micro[i][-1]], 1))
                prior_mean_t = self.prior_mean[i](prior_t)
                prior_std_t = self.prior_std[i](prior_t)

                z_t = sample_gauss(prior_mean_t, prior_std_t)

                dec_t = self.dec[i](torch.cat([y_t, m_t, z_t, h_micro[i][-1]], 1))
                dec_mean_t = self.dec_mean[i](dec_t)
                dec_std_t = self.dec_std[i](dec_t)

                if t >= burn_in:
                    states[t+1,:,2*i:2*i+2] = sample_gauss(dec_mean_t, dec_std_t)
                    
                _, h_micro[i] = self.gru_micro[i](torch.cat([states[t+1,:,2*i:2*i+2], z_t], 1).unsqueeze(0), h_micro[i])

        macro_intents.data[-1] = macro_intents.data[-2]

        return states, macro_intents
