import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable

from bball_data import BBallData
from model import *


def printlog(line):
	print(line)
	with open(save_path+'log.txt', 'a') as file:
		file.write(line+'\n')


def loss_str(losses):
	ret = ''
	for key in losses:
		ret += ' {}: {:.4f} |'.format(key, losses[key])
	ret += ' total_loss: {:.4f}'.format(sum(losses.values()))
	return ret


def hyperparams_str(epoch, hp):
	ret = 'Epoch: {:d}'.format(epoch)

	if hp['pretrain']:
		ret += ' (pretrain)'
	if warmup > 0:
		ret += ' | Beta: {:.2f}'.format(hp['beta'])
	if min_eps < 1 or eps_start < n_epochs:
		ret += ' | Epsilon: {:.2f}'.format(hp['eps'])
	if 'GUMBEL' in params['model']:
		ret += ' | Tau: {:.2f}'.format(hp['tau'])

	return ret


def run_epoch(train, hp):
	loader = train_loader if train else test_loader
	losses = {}

	for batch_idx, (data, macro_goals) in enumerate(loader):
		if args.cuda:
			data, macro_goals = data.cuda(), macro_goals.cuda()

		# change (batch, time, x) to (time, batch, x)
		data = Variable(data.squeeze().transpose(0, 1))
		macro_goals = Variable(macro_goals.squeeze().transpose(0, 1))

		batch_losses = model(data, macro_goals, hp)

		if train:
			optimizer.zero_grad()
			total_loss = sum(batch_losses.values())
			total_loss.backward()
			nn.utils.clip_grad_norm(model.parameters(), clip)
			optimizer.step()
		
		for key in batch_losses:
			if batch_idx == 0:
				losses[key] = batch_losses[key].data[0]
			else:
				losses[key] += batch_losses[key].data[0]

	for key in losses:
		losses[key] /= len(loader.dataset)

	return losses


######################################################################
######################### MAIN STARTS HERE ###########################
######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--x_dim', type=int, required=True)
parser.add_argument('--y_dim', type=int, required=True)
parser.add_argument('--z_dim', type=int, required=True)
parser.add_argument('--h_dim', type=int, required=True, help='hidden state dimension')
parser.add_argument('--m_dim', type=int, required=True, help='macro-goal dimension')
parser.add_argument('--rnn_dim', type=int, required=True, help='num recurrent cells for next action/state')
parser.add_argument('--rnn_micro_dim', type=int, required=True, help='same as rnn_dim for macro-goal models')
parser.add_argument('--rnn_macro_dim', type=int, required=True, help='num recurrent cells for macro-goals')
parser.add_argument('--n_agents', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=False, default=1, help='num layers in recurrent cells')
parser.add_argument('--subsample', type=int, required=False, default=1, help='subsample sequeneces')
parser.add_argument('--seed', type=int, required=False, default=128, help='PyTorch random seed')
parser.add_argument('--n_epochs', type=int, required=True)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--start_lr', type=float, required=True, help='starting learning rate')
parser.add_argument('--min_lr', type=float, required=True, help='minimum learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--min_eps', type=float, required=False, default=1, help='minimum epsilon for scheduled sampling')
parser.add_argument('--warmup', type=int, required=False, default=0, help='warmup for KL term')
parser.add_argument('--pretrain', type=int, required=False, default=50, help='num epochs to train macro-goal policy')
parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
parser.add_argument('--cont', action='store_true', default=False, help='continue training a model')
args = parser.parse_args()

if not torch.cuda.is_available():
	args.cuda = False

# model parameters
params = {
	'model' : args.model,
	'genMacro' : (args.model[:5]=='MACRO'),
	'x_dim' : args.x_dim,
	'y_dim' : args.y_dim,
	'z_dim' : args.z_dim,
	'h_dim' : args.h_dim,
	'm_dim' : args.m_dim,
	'rnn_dim' : args.rnn_dim,
	'rnn_micro_dim' : args.rnn_micro_dim,
	'rnn_macro_dim' : args.rnn_macro_dim,
	'n_agents' : args.n_agents,
	'n_layers' : args.n_layers,
	'subsample' : args.subsample,
	'seed' : args.seed,
	'cuda' : args.cuda
}

# hyperparameters
n_epochs = args.n_epochs
clip = args.clip
start_lr = args.start_lr
min_lr = args.min_lr
batch_size = args.batch_size
save_every = args.save_every

# scheduled sampling
min_eps = args.min_eps
eps_start = n_epochs

# anneal KL term in loss
warmup = args.warmup

# multi-stage training
pretrain_time = args.pretrain if params['genMacro'] else 0 

# set manual seed
torch.manual_seed(params['seed'])
if args.cuda:
	torch.cuda.manual_seed(params['seed'])

# load model
model = eval(params['model'])(params)
if args.cuda:
	model.cuda()
params['total_params'] = num_trainable_params(model)
print(params)

# create save path and saving parameters
save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
	os.makedirs(save_path)
	os.makedirs(save_path+'model/')
pickle.dump(params, open(save_path+'params.p', 'wb'), protocol=2)

# continue a previous experiment, but currently have to manually choose model
if args.cont:
	state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best.pth')
	model.load_state_dict(state_dict)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
	BBallData(train=True, preprocess=True, subsample=params['subsample']), 
	batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	BBallData(train=False, preprocess=True, subsample=params['subsample']), 
	batch_size=batch_size, shuffle=True, **kwargs)


best_test_loss = 0
lr = start_lr

for e in range(n_epochs):
	epoch = e+1

	hyperparams = {
		'beta' : 1 if epoch > warmup else epoch/warmup,
		'eps' : 0 if epoch < eps_start else int((epoch-eps_start)/10) + 1,
		'tau' : max(2.5*math.exp(-e/100), 0.1),
		'pretrain' : epoch <= pretrain_time
	}

	# can set a custom learning rate schedule
	# filter removes parameters with requires_grad=False
	# https://github.com/pytorch/pytorch/issues/679
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=lr)

	printlog(hyperparams_str(epoch, hyperparams))	
	start_time = time.time()

	train_loss = run_epoch(train=True, hp=hyperparams)
	printlog('Train:\t' + loss_str(train_loss))

	test_loss = run_epoch(train=False, hp=hyperparams)
	printlog('Test:\t' + loss_str(test_loss))

	epoch_time = time.time() - start_time
	printlog('Time:\t {:.3f}'.format(epoch_time))

	total_test_loss = sum(test_loss.values())

	# best model on test set
	if best_test_loss == 0 or total_test_loss < best_test_loss:	
		best_test_loss = total_test_loss
		filename = save_path+'model/'+params['model']+'_state_dict_best.pth'

		if epoch <= pretrain_time:
			filename = save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth'

		torch.save(model.state_dict(), filename)
		printlog('Best model at epoch '+str(epoch))

	# periodically save model
	if epoch % save_every == 0:
		filename = save_path+'model/'+params['model']+'_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), filename)
		printlog('Saved model')

	# end of pretrain stage
	if epoch == pretrain_time:
		printlog('END of pretrain')
		best_test_loss = 0
		lr = start_lr

		state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth')
		model.load_state_dict(state_dict)

		test_loss = run_epoch(train=False, hp=hyperparams)
		printlog('Test:\t' + loss_str(test_loss))

printlog('Best Test Loss: {:.4f}'.format(best_test_loss))