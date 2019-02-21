import torch
import math


######################################################################
############################ MODEL UTILS #############################
######################################################################


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        parser.add_argument('--'+arg, type=int, required=True)
    args, _ = parser.parse_known_args()

    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ''
    for arg in model_args:
        ret += ' {} {} |'.format(arg, params[arg])
    return ret[1:-2]


def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states


def index_by_agent(states, n_agents):
    x = states[1:,:,:2*n_agents].clone()
    x = x.view(x.size(0), x.size(1), n_agents, -1).transpose(1,2)
    return x


def get_macro_ohe(macro, n_agents, M):
    macro_ohe = torch.zeros(macro.size(0), n_agents, macro.size(1), M)
    for i in range(n_agents):
        macro_ohe[:,i,:,:] = one_hot_encode(macro[:,:,i].data, M)
    if macro.is_cuda:
        macro_ohe = macro_ohe.cuda()

    return macro_ohe


######################################################################
############################## GAUSSIAN ##############################
######################################################################


def sample_gauss(mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(std).add_(mean)


def nll_gauss(mean, std, x):
    pi = torch.FloatTensor([math.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    nll_element = (x - mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
    
    return 0.5 * torch.sum(nll_element)
    

def kld_gauss(mean_1, std_1, mean_2, std_2):
    kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    return  0.5 * torch.sum(kld_element)


def entropy_gauss(std, scale=1):
    """Computes gaussian differential entropy."""
    pi, e = torch.FloatTensor([math.pi]), torch.FloatTensor([math.e])
    if std.is_cuda:
        pi, e = pi.cuda(), e.cuda()
    return 0.5 * torch.sum(scale*torch.log(2*pi*e*std))


######################################################################
########################## MISCELLANEOUS #############################
######################################################################


def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).cpu().long()
    dims.append(N)
    ret = torch.zeros(dims)
    ret.scatter_(-1, inds, 1)
    return ret


def sample_multinomial(probs):
    inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    if probs.is_cuda:
        ret = ret.cuda()
    return ret
