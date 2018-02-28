import argparse
import os
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *

from bball_data import BBallData


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True, help='trial')
parser.add_argument('-n', '--n_samples', type=int, default=5, required=False, help='number of samples')
parser.add_argument('-b', '--burn_in', type=int, default=0, required=False, help='burn-in period')
parser.add_argument('-l', '--seq_len', type=int, default=0, required=False, help='length of sequence')
parser.add_argument('-m', '--model', type=str, default='best', required=False, help='which saved model to sample from')
parser.add_argument('-f', '--filedesc', type=str, default='', required=False, help='descriptor to add to end of filename')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle ground-truth burn-in from test set')
args = parser.parse_args()	


trial = args.trial
save_path = 'saved/%03d/' % trial
params = pickle.load(open(save_path+'params.p', 'rb'))

# make samples folder
if not os.path.exists(save_path+'samples/'):
	os.makedirs(save_path+'samples/')

# load the model
state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_'+args.model+'.pth')
model = eval(params['model'])(params)
if params['cuda']:
	model.cuda()
model.load_state_dict(state_dict)

# set the burn-in (and save for plotting)
# TODO: need a better way to save different burn-ins for different sets of samples
params['burn_in'] = args.burn_in
pickle.dump(params, open(save_path+'params.p', 'wb'), protocol=2)
print(params)

# set up the file name
file_desc = '' if len(args.filedesc) == 0 else '_'+args.filedesc

# sample for a fixed sequence length
if args.seq_len > 0:
	file_desc += '_len'+str(args.seq_len)
	
# load ground-truth burn-ins
test_loader = torch.utils.data.DataLoader(
	BBallData(train=False, preprocess=True, subsample=params['subsample']),
	batch_size=args.n_samples, shuffle=args.shuffle)

data, macro_goals = next(iter(test_loader))
if params['cuda']:
		data, macro_goals = data.cuda(), macro_goals.cuda()

data = Variable(data.squeeze().transpose(0, 1))
macro_goals = Variable(macro_goals.squeeze().transpose(0, 1))

# generate samples
if params.get('genMacro'):
	samples, macro_samples = model.sample(data, macro_goals, burn_in=params['burn_in'], seq_len=args.seq_len)

	# save macro-goals
	macro_samples = macro_samples.data.cpu().numpy()
	pickle.dump(macro_samples, open(save_path+'samples/macro_goals'+file_desc+'.p', 'wb'))
else:
	samples = model.sample(data, macro_goals, burn_in=params['burn_in'], seq_len=args.seq_len)

# save samples
samples = samples.data.cpu().numpy()
pickle.dump(samples, open(save_path+'samples/samples'+file_desc+'.p', 'wb'))