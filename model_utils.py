import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


######################################################################
########################## MISCELLANEOUS #############################
######################################################################


def one_hot_encode(inds, N):
	# inds should be a torch.Tensor, not a Variable
	dims = [inds.size(i) for i in range(len(inds.size()))]
	inds = inds.unsqueeze(-1).cpu().long()
	dims.append(N)
	ret = torch.zeros(dims)
	ret.scatter_(-1, inds, 1)
	return ret


def logsumexp(x, axis=None):
    x_max = torch.max(x, axis, keepdim=True)[0] # torch.max() returns a tuple
    ret = torch.log(torch.sum(torch.exp(x - x_max), axis, keepdim=True)) + x_max
    return ret


######################################################################
############################ SAMPLING ################################
######################################################################


def sample_gumbel(logits, tau=1, eps=1e-20):
	u = torch.zeros(logits.size()).uniform_()
	u = Variable(u)
	if logits.is_cuda:
		u = u.cuda()
	g = -torch.log(-torch.log(u+eps)+eps)
	y = (g+logits) / tau
	return F.softmax(y)


def sample_gauss(mean, std):
	eps = torch.FloatTensor(std.size()).normal_()
	eps = Variable(eps)
	if mean.is_cuda:
		eps = eps.cuda()
	return eps.mul(std).add_(mean)


def sample_gmm(mean, std, coeff):
	k = coeff.size(-1)
	if k == 1:
		return sample_gauss(mean, std)
	
	mean = mean.view(mean.size(0), -1, k)
	std = std.view(std.size(0), -1, k)
	index = torch.multinomial(coeff,1).squeeze()

	# TODO: replace with torch.gather or torch.index_select
	comp_mean = Variable(torch.zeros(mean.size()[:-1]))
	comp_std = Variable(torch.zeros(std.size()[:-1]))
	if mean.is_cuda:
		comp_mean = comp_mean.cuda()
		comp_std = comp_std.cuda()
	for i in range(index.size(0)):
		comp_mean[i,:] = mean.data[i,:,index.data[i]]
		comp_std[i,:] = std.data[i,:,index.data[i]]	
	
	return sample_gauss(comp_mean, comp_std), index


def sample_multinomial(probs):
	inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
	ret = one_hot_encode(inds, probs.size(-1))
	if probs.is_cuda:
		ret = ret.cuda()
	return ret


######################################################################
######################### KL DIVERGENCE ##############################
######################################################################


def kld_gauss(mean_1, std_1, mean_2, std_2):
	kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
		(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
		std_2.pow(2) - 1)
	return	0.5 * torch.sum(kld_element)


def kld_categorical(logits_1, logits_2):
	kld_element = torch.exp(logits_1) * (logits_1 - logits_2)
	return torch.sum(kld_element)


######################################################################
###################### NEGATIVE LOG-LIKELIHOOD #######################
######################################################################


def nll_gauss(mean, std, x):
	pi = Variable(torch.FloatTensor([np.pi]))
	if mean.is_cuda:
		pi = pi.cuda()
	nll_element = (x - mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
	
	return 0.5 * torch.sum(nll_element)


def nll_gmm(mean, std, coeff, x):
	# mean: (batch, x_dim*k)
	# std: (batch, x_dim*k)
	# coeff: (batch, k)
	# x: (batch, x_dim)

	k = coeff.size(-1)
	if k == 1:
		return nll_gauss(mean, std, x)

	pi = Variable(torch.FloatTensor([np.pi]))
	if mean.is_cuda:
		pi = pi.cuda()
	mean = mean.view(mean.size(0), -1, k)
	std = std.view(std.size(0), -1, k)

	nll_each = (x.unsqueeze(-1) - mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
	nll_component = -0.5 * torch.sum(nll_each, 1)
	terms = torch.log(coeff) + nll_component

	return -torch.sum(logsumexp(terms, axis=1))


######################################################################
###################### METHODS FOR LOG-VARIANCE ######################
######################################################################


def sample_gauss_logvar(mean, logvar):
	eps = torch.FloatTensor(mean.size()).normal_()
	eps = Variable(eps)
	if mean.is_cuda:
		eps = eps.cuda()
	return eps.mul(torch.exp(logvar/2)).add_(mean)


def kld_gauss_logvar(mean_1, logvar_1, mean_2, logvar_2):
	kld_element =  (logvar_2 - logvar_1 + 
		(torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) /
		torch.exp(logvar_2) - 1)
	return	0.5 * torch.sum(kld_element)


def nll_gauss_logvar(mean, logvar, x):
	pi = Variable(torch.FloatTensor([np.pi]))
	if mean.is_cuda:
		pi = pi.cuda()
	nll_element = (x - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2*pi)
	
	return 0.5 * torch.sum(nll_element)