import argparse
import os

import sys
sys.path.append(sys.path[0] + '/..')

from torch.utils.data import DataLoader
from importlib import import_module
from datasets import GeneralDataset


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='bball', required=False, help='dataset')
parser.add_argument('-n', '--n_samples', type=int, default=5, required=False, help='number of samples')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle ground-truth burn-in from test set')
parser.add_argument('--animate', action='store_true', default=False, help='animate sequences')
args, _ = parser.parse_known_args()


# Create save destination
save_path = 'datasets/{}/data/examples'.format(args.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set params
params = {
    'dataset' : args.dataset,
    'normalize' : True,
    'n_samples' : args.n_samples,
    'burn_in' : 0,
    'genMacro' : True
}   

# Load ground-truth states from test set
test_loader = DataLoader(
    GeneralDataset(params['dataset'], train=False, normalize_data=params['normalize'], subsample=1), 
    batch_size=args.n_samples, shuffle=args.shuffle)
data, macro_intents = next(iter(test_loader))
data, macro_intents = data.detach().numpy(), macro_intents.detach().numpy()

# Get dataset plot function
dataset = import_module('datasets.{}'.format(params['dataset']))
plot_func = dataset.animate if args.animate else dataset.display

for k in range(args.n_samples):
    print('Sample {:02d}'.format(k))
    save_file = '{}/{:02d}'.format(save_path, k)
    plot_func(data[k], macro_intents[k], params=params, save_file=save_file)
    