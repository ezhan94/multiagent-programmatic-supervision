import torch
import torch.nn as nn

from models.utils import parse_model_params, get_params_str
from models.utils import sample_gauss, nll_gauss, kld_gauss


class VRNN_MIXED(nn.Module):
    """
    Combination of VRNN-single and VRNN-indep.
    Single RNN for all agents, but separate decoders.
    """

    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'rnn_dim', 'n_layers', 'n_agents']
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        x_dim = params['x_dim']
        y_dim = params['y_dim']
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        rnn_dim = params['rnn_dim']
        n_layers = params['n_layers']
        n_agents = params['n_agents']

        self.enc = nn.ModuleList([nn.Sequential(
            nn.Linear(x_dim+y_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.enc_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.enc_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.prior = nn.ModuleList([nn.Sequential(
            nn.Linear(y_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.prior_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.prior_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(y_dim+z_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_agents)])
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.rnn = nn.GRU(y_dim+n_agents*z_dim, rnn_dim, n_layers)


    def forward(self, states, macro=None, hp=None):
        out = {}
        out['kl_loss'] = 0
        out['recon_loss'] = 0

        n_agents = self.params['n_agents']

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])
        if self.params['cuda']:
            h = h.cuda()

        for t in range(states.size(0)-1):
            y_t = states[t].clone()

            z_list = []

            for i in range(n_agents):
                x_t = states[t+1][:,2*i:2*i+2].clone()

                enc_t = self.enc[i](torch.cat([x_t, y_t, h[-1]], 1))
                enc_mean_t = self.enc_mean[i](enc_t)
                enc_std_t = self.enc_std[i](enc_t)

                prior_t = self.prior[i](torch.cat([y_t, h[-1]], 1))
                prior_mean_t = self.prior_mean[i](prior_t)
                prior_std_t = self.prior_std[i](prior_t)

                z_t = sample_gauss(enc_mean_t, enc_std_t)
                z_list.append(z_t)

                dec_t = self.dec[i](torch.cat([y_t, z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean[i](dec_t)
                dec_std_t = self.dec_std[i](dec_t)

                out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

            z_concat = torch.cat(z_list, -1)
            _, h = self.rnn(torch.cat([y_t, z_concat], 1).unsqueeze(0), h)

        return out
    

    def sample(self, states, macro=None, burn_in=0):
        n_agents = self.params['n_agents']

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])

        for t in range(states.size(0)-1):
            y_t = states[t].clone()

            z_list = []

            for i in range(n_agents):
                prior_t = self.prior[i](torch.cat([y_t, h[-1]], 1))
                prior_mean_t = self.prior_mean[i](prior_t)
                prior_std_t = self.prior_std[i](prior_t)

                z_t = sample_gauss(prior_mean_t, prior_std_t)
                z_list.append(z_t)

                dec_t = self.dec[i](torch.cat([y_t, z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean[i](dec_t)
                dec_std_t = self.dec_std[i](dec_t)

                if t >= burn_in:
                    states[t+1,:,2*i:2*i+2] = sample_gauss(dec_mean_t, dec_std_t)

            z_concat = torch.cat(z_list, -1)
            _, h = self.rnn(torch.cat([y_t, z_concat], 1).unsqueeze(0), h)  

        return states, None
