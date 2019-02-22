import argparse
import os
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
from importlib import import_module
from datasets import GeneralDataset
from models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True, help='trial/model')
parser.add_argument('-n', '--n_samples', type=int, default=5, required=False, help='number of samples')
parser.add_argument('-b', '--burn_in', type=int, default=0, required=False, help='burn-in period')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle ground-truth burn-in from test set')
parser.add_argument('--run', action='store_true', default=False, help='generate new samples')
parser.add_argument('--plot', action='store_true', default=False, help='plot and save samples')
parser.add_argument('--animate', action='store_true', default=False, help='animate and save samples')
args, _ = parser.parse_known_args()


# Make experiment folder
model_path ='saved/{:03d}'.format(args.trial)
experiment_path = '{}/experiments/sample'.format(model_path)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

if args.run:
    # Load params
    params = pickle.load(open(model_path+'/params.p', 'rb'))

    # Load model
    state_dict = torch.load('{}/model/{}_state_dict_best.pth'.format(model_path, params['model']), map_location=lambda storage, loc: storage)
    model = load_model(params['model'], params)
    model.load_state_dict(state_dict)

    # Load ground-truth states from test set
    test_loader = DataLoader(
        GeneralDataset(params['dataset'], train=False, normalize_data=params['normalize'], subsample=params['subsample']), 
        batch_size=args.n_samples, shuffle=args.shuffle)
    data, macro_intents = next(iter(test_loader))
    data, macro_intents = data.transpose(0, 1), macro_intents.transpose(0, 1)
    
    # Sample trajectories
    samples, macro_samples = model.sample(data, macro_intents, burn_in=args.burn_in)

    # Save samples
    samples = samples.detach().numpy()
    pickle.dump(samples, open(experiment_path+'/samples.p', 'wb'), protocol=2)

    # Save macro_intents
    if macro_samples is not None:
        macro_samples = macro_samples.detach().numpy()
        pickle.dump(macro_samples, open(experiment_path+'/macro_intents.p', 'wb'), protocol=2)

    # Save experiment parameters
    exp_params = {
        'dataset' : params['dataset'],
        'normalize' : params['normalize'],
        'n_samples' : args.n_samples,
        'burn_in' : args.burn_in,
        'genMacro' : macro_samples is not None
    }   
    pickle.dump(exp_params, open(experiment_path+'/exp_params.p', 'wb'), protocol=2)

if args.plot:
    # Create save destination
    save_path = experiment_path+'/plots'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load experiment parameters
    exp_params = pickle.load(open(experiment_path+'/exp_params.p', 'rb'))

    # Load samples
    samples = pickle.load(open(experiment_path+'/samples.p', 'rb'))
    samples = np.swapaxes(samples, 0, 1)

    # Load macro-intents
    macro_intents = [None]*len(samples)
    if exp_params['genMacro']:
        macro_intents = pickle.load(open(experiment_path+'/macro_intents.p', 'rb'))
        macro_intents = np.swapaxes(macro_intents, 0, 1)

    # Get dataset plot function
    dataset = import_module('datasets.{}'.format(exp_params['dataset']))
    plot_func = dataset.animate if args.animate else dataset.display

    for k in range(exp_params['n_samples']):
        print('Sample {:02d}'.format(k))
        save_file = '{}/{:02d}'.format(save_path, k)
        plot_func(samples[k], macro_intents[k], params=exp_params, save_file=save_file)
    