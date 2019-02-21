import torch
import torch.nn as nn

from models.utils import parse_model_params, get_params_str
from models.utils import sample_gauss, nll_gauss


class RNN_GAUSS(nn.Module):
    """RNN with Gaussian output distribution."""

    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['x_dim', 'y_dim', 'h_dim', 'rnn_dim', 'n_layers']
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        x_dim = params['x_dim']
        y_dim = params['y_dim']
        h_dim = params['h_dim']
        rnn_dim = params['rnn_dim']
        n_layers = params['n_layers']

        self.dec = nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Linear(h_dim, y_dim)
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.rnn = nn.GRU(y_dim, rnn_dim, n_layers)


    def forward(self, states, macro=None, hp=None):
        out = {}
        out['nll'] = 0

        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])
        if self.params['cuda']:
            h = h.cuda()            
        
        for t in range(states.size(0)):
            y_t = states[t].clone()

            dec_t = self.dec(torch.cat([h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            _, h = self.rnn(y_t.unsqueeze(0), h)

            out['nll'] += nll_gauss(dec_mean_t, dec_std_t, y_t)

        return out
    
    
    def sample(self, states, macro=None, burn_in=0):
        h = torch.zeros(self.params['n_layers'], states.size(1), self.params['rnn_dim'])

        for t in range(states.size(0)):
            dec_t = self.dec(torch.cat([h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            if t >= burn_in:
                states[t] = sample_gauss(dec_mean_t, dec_std_t)

            _, h = self.rnn(states[t].unsqueeze(0), h)

        return states, None
