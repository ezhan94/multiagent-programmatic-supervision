import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from bball_data.utils import unnormalize, plot_sequence, animate_sequence


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True, help='trial')
parser.add_argument('-f', '--filedesc', type=str, default='', required=False, help='descriptor to add to end of filename')
parser.add_argument('--animate', action='store_true', default=False, help='animate sequences')
parser.add_argument('--showmacro', action='store_true', default=False, help='show macro-goals')
parser.add_argument('--save', action='store_true', default=False, help='save plots')
parser.add_argument('--params', action='store_true', default=False, help='just print params')
args = parser.parse_args()	

trial = args.trial
load_path = 'saved/%03d/' % trial
save_path = load_path+'plots/'

# create save folder
if args.save and not os.path.exists(save_path):
	os.makedirs(save_path)

# display model parameters
params = pickle.load(open(load_path+'params.p', 'rb'))
print(params)
if args.params:
	quit()

# load samples
file_desc = '' if len(args.filedesc) == 0 else '_'+args.filedesc
samples = pickle.load(open(load_path+'samples/samples'+file_desc+'.p', 'rb'))
samples = np.swapaxes(samples, 0, 1)

# load macro-goals
macro_goals = [None]*len(samples)
if params.get('genMacro') and args.showmacro:
	macro_goals = pickle.load(open(load_path+'samples/macro_goals'+file_desc+'.p', 'rb'))
	macro_goals = np.swapaxes(macro_goals, 0, 1)

# plot samples (and save)
for k, sample in enumerate(samples):
	print('Sample %02d' % k)
	sample = unnormalize(sample)
	macro = macro_goals[k]
	save_name = '%02d' % k if args.save else ''

	if not args.animate:
		plot_sequence(sample, macro_goals=macro, burn_in=params['burn_in'], save_path=save_path, save_name=save_name)	
	else:
		animate_sequence(sample, macro_goals=macro, burn_in=params['burn_in'], save_path=save_path, save_name=save_name)