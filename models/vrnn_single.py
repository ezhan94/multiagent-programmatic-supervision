import torch
import torch.nn as nn

from models.utils import parse_model_params, get_params_str
from models.utils import sample_gauss, nll_gauss, kld_gauss


class VRNN_SINGLE(nn.Module):
    """Single VRNN model for all agents concatenated together."""

    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['y_dim', 'z_dim', 'h_dim', 'rnn_dim', 'n_layers']
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        y_dim = params['y_dim']
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        rnn_dim = params['rnn_dim']
        n_layers = params['n_layers']

        self.enc = nn.Sequential(
            nn.Linear(y_dim+y_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.prior = nn.Sequential(
            nn.Linear(y_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.dec = nn.Sequential(
            nn.Linear(y_dim+z_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Linear(h_dim, y_dim)
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.rnn = nn.GRU(y_dim+z_dim, rnn_dim, n_layers)


    def forward(self, states, macro=None, hp=None):
        out = {}
        out['kl_loss'] = 0
        out['recon_loss'] = 0

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])
        if self.params['cuda']:
            h = h.cuda()            
        
        for t in range(states.size(0)-1):

            y_t = states[t].clone()
            x_t = states[t+1].clone()

            enc_t = self.enc(torch.cat([x_t, y_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            prior_t = self.prior(torch.cat([y_t, h[-1]], 1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            z_t = sample_gauss(enc_mean_t, enc_std_t)

            dec_t = self.dec(torch.cat([y_t, z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            _, h = self.rnn(torch.cat([x_t, z_t], 1).unsqueeze(0), h)

            out['kl_loss'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

        return out
    

    def sample(self, states, macro=None, burn_in=0):
        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])

        for t in range(states.size(0)-1):
            y_t = states[t].clone()
            
            prior_t = self.prior(torch.cat([y_t, h[-1]], 1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            z_t = sample_gauss(prior_mean_t, prior_std_t)

            dec_t = self.dec(torch.cat([y_t, z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            if t >= burn_in:
                states[t+1] = sample_gauss(dec_mean_t, dec_std_t)

            _, h = self.rnn(torch.cat([states[t+1], z_t], 1).unsqueeze(0), h)

        return states, None
