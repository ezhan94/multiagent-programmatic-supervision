import torch
import torch.nn as nn

from models.utils import parse_model_params, get_params_str
from models.utils import sample_gauss, nll_gauss, kld_gauss, entropy_gauss


class VRAE_MI(nn.Module):
    """
    VRAE-style architecture from https://arxiv.org/abs/1412.6581
    We maximize the mutual informatio between the latent variables and the trajectories.
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

        self.enc = nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(z_dim+rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()) for i in range(n_agents)])
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_agents)])
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.discrim = nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.discrim_mean = nn.Linear(h_dim, z_dim)
        self.discrim_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.rnn = nn.GRU(y_dim, rnn_dim, n_layers)


    def forward(self, states, macro=None, hp=None):
        out = {}
        out['kl_loss'] = 0
        out['recon_loss'] = 0
        out['z_entropy'] = 0
        out['discrim_loss'] = 0

        n_agents = self.params['n_agents']

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])
        if self.params['cuda']:
            h = h.cuda()

        for t in range(states.size(0)):
            y_t = states[t].clone()
            _, h = self.rnn(y_t.unsqueeze(0), h)

        enc = self.enc(h[-1])
        enc_mean = self.enc_mean(enc)
        enc_std = self.enc_std(enc)

        z = sample_gauss(enc_mean, enc_std)
        prior_mean = torch.zeros(enc_mean.size()).to(enc_mean.device)
        prior_std = torch.ones(enc_std.size()).to(enc_std.device)

        out['kl_loss'] += kld_gauss(enc_mean, enc_std, prior_mean, prior_std)
        out['z_entropy'] -= entropy_gauss(enc_std)

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])
        if self.params['cuda']:
            h = h.cuda()

        for t in range(states.size(0)-1):
            y_t = states[t].clone()
            _, h = self.rnn(y_t.unsqueeze(0), h)

            for i in range(n_agents):
                x_t = states[t+1][:,2*i:2*i+2].clone()

                dec_t = self.dec[i](torch.cat([z, h[-1]], 1))
                dec_mean_t = self.dec_mean[i](dec_t)
                dec_std_t = self.dec_std[i](dec_t)

                out['recon_loss'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

        discrim = self.discrim(h[-1])
        discrim_mean = self.discrim_mean(discrim)
        discrim_std = self.discrim_std(discrim)

        out['discrim_loss'] += nll_gauss(discrim_mean, discrim_std, z)

        return out
    

    def sample(self, states, macro=None, burn_in=0):
        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])

        prior_mean = torch.zeros(states.size(1), self.params['z_dim']).to(states.device)
        prior_std = torch.ones(states.size(1), self.params['z_dim']).to(states.device)
        z = sample_gauss(prior_mean, prior_std)

        for t in range(states.size(0)-1):
            y_t = states[t].clone()
            _, h = self.rnn(y_t.unsqueeze(0), h)

            for i in range(self.params['n_agents']):
                dec_t = self.dec[i](torch.cat([z, h[-1]], 1))
                dec_mean_t = self.dec_mean[i](dec_t)
                dec_std_t = self.dec_std[i](dec_t)

                if t >= burn_in:
                    states[t+1,:,2*i:2*i+2] = sample_gauss(dec_mean_t, dec_std_t)

        return states, None
