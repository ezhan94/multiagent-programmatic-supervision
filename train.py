import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import GeneralDataset
from models import load_model
from models.utils import num_trainable_params


#################################################
############### HELPER FUNCTIONS ################
#################################################


def printlog(line):
    print(line)
    with open(save_path+'/log.txt', 'a') as file:
        file.write(line+'\n')


def loss_str(losses):
    ret = ''
    for key in losses:
        ret += ' {}: {:.4f} |'.format(key, losses[key])
    if len(losses) > 1:
        ret += ' total_loss: {:.4f} |'.format(sum(losses.values()))
    return ret[:-2]


def hyperparams_str(epoch, hp):
    ret = 'Epoch {:d}'.format(epoch)

    if hp['pretrain']:
        ret += ' (pretrain)'

    return ret


#################################################
################### ONE EPOCH ###################
#################################################


def run_epoch(train, hp):
    loader = train_loader if train else test_loader
    losses = {}

    for batch_idx, (data, macro_intents) in enumerate(loader):
        if args.cuda:
            data, macro_intents = data.cuda(), macro_intents.cuda()

        # Change (batch, time, x) to (time, batch, x)
        data = data.transpose(0, 1)
        macro_intents = macro_intents.transpose(0, 1)

        batch_losses = model(data, macro_intents, hp)

        if train:
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item()
            else:
                losses[key] += batch_losses[key].item()

    for key in losses:
        losses[key] /= len(loader.dataset)

    return losses


######################################################################
######################### MAIN STARTS HERE ###########################
######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--n_epochs', type=int, required=False, default=200, help='num epochs')
parser.add_argument('--min_lr', type=float, required=False, default=0.0001, help='minimum learning rate')
parser.add_argument('--start_lr', type=float, required=False, default=0.0001, help='starting learning rate')
parser.add_argument('--subsample', type=int, required=False, default=1, help='subsample sequence')
parser.add_argument('--clip', type=int, required=False, default=10, help='gradient clipping')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='batch size')
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--seed', type=int, required=False, default=128, help='PyTorch random seed')
parser.add_argument('--normalize', action='store_true', default=True, help='normalize data')
parser.add_argument('--pretrain_time', type=int, required=False, default=0, help='num epochs to train macro-intent policy')
parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
parser.add_argument('--cont', action='store_true', default=False, help='continue training previous best model')
args, _ = parser.parse_known_args()

if not torch.cuda.is_available():
    args.cuda = False

# Parameters to save
params = {
    'model' : args.model,
    'dataset' : args.dataset,
    'min_lr' : args.min_lr,
    'start_lr' : args.start_lr,
    'subsample' : args.subsample,
    'normalize' : args.normalize,
    'seed' : args.seed,
    'cuda' : args.cuda
}

# Hyperparameters
n_epochs = args.n_epochs
clip = args.clip
batch_size = args.batch_size
save_every = args.save_every
pretrain_time = args.pretrain_time

# Set manual seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load model
model = load_model(args.model, params, parser)
if args.cuda:
    model.cuda()

# Update params with model parameters
params = model.params
params['total_params'] = num_trainable_params(model)

# Create save path and saving parameters
save_path = 'saved/{:03d}'.format(args.trial)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'/model')
pickle.dump(params, open(save_path+'/params.p', 'wb'), protocol=2)

# Continue a previous experiment, or start a new one
if args.cont:
    state_dict = torch.load('{}/model/{}_state_dict_best.pth'.format(save_path, args.model))
    model.load_state_dict(state_dict)
else:
    printlog('{:03d} {} {}'.format(args.trial, args.model, args.dataset))
    printlog(model.params_str)
    printlog('start_lr {} | min_lr {} | subsample {} | batch_size {} | seed {}'.format(
        args.start_lr, args.min_lr, args.subsample, args.batch_size, args.seed))
    printlog('n_params: {:,}'.format(params['total_params']))
    printlog('best_loss:')
printlog('############################################################')

# Dataset loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = DataLoader(
    GeneralDataset(args.dataset, train=True, normalize_data=args.normalize, subsample=args.subsample),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    GeneralDataset(args.dataset, train=False, normalize_data=args.normalize, subsample=args.subsample),
    batch_size=batch_size, shuffle=True, **kwargs)

############################# TRAIN LOOP #############################

best_test_loss = 0
epochs_since_best = 0
lr = max(args.start_lr, args.min_lr)

for e in range(n_epochs):
    epoch = e+1

    hyperparams = {
        'pretrain' : epoch <= pretrain_time
    }

    # Set a custom learning rate schedule
    if epochs_since_best == 5 and lr > args.min_lr:
        # Load previous best model
        filename = '{}/model/{}_state_dict_best.pth'.format(save_path, args.model)
        if epoch <= pretrain_time:
            filename = '{}/model/{}_state_dict_best_pretrain.pth'.format(save_path, args.model)
        state_dict = torch.load(filename)

        # Decrease learning rate
        lr = max(lr/3, args.min_lr)
        printlog('########## lr {} ##########'.format(lr))
        epochs_since_best = 0
    else:
        epochs_since_best += 1

    # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    printlog(hyperparams_str(epoch, hyperparams))   
    start_time = time.time()

    train_loss = run_epoch(train=True, hp=hyperparams)
    printlog('Train:\t'+loss_str(train_loss))

    test_loss = run_epoch(train=False, hp=hyperparams)
    printlog('Test:\t'+loss_str(test_loss))

    epoch_time = time.time() - start_time
    printlog('Time:\t {:.3f}'.format(epoch_time))

    total_test_loss = sum(test_loss.values())

    # Best model on test set
    if best_test_loss == 0 or total_test_loss < best_test_loss: 
        best_test_loss = total_test_loss
        epochs_since_best = 0

        filename = '{}/model/{}_state_dict_best.pth'.format(save_path, args.model)
        if epoch <= pretrain_time:
            filename = '{}/model/{}_state_dict_best_pretrain.pth'.format(save_path, args.model)

        torch.save(model.state_dict(), filename)
        printlog('##### Best model #####')

    # Periodically save model
    if epoch % save_every == 0:
        filename = '{}/model/{}_state_dict_{}.pth'.format(save_path, args.model, epoch)
        torch.save(model.state_dict(), filename)
        printlog('########## Saved model ##########')

    # End of pretrain stage
    if epoch == pretrain_time:
        printlog('########## END pretrain ##########')
        best_test_loss = 0
        epochs_since_best = 0
        lr = max(args.start_lr, args.min_lr)

        state_dict = torch.load('{}/model/{}_state_dict_best_pretrain.pth'.format(save_path, args.model))
        model.load_state_dict(state_dict)

        test_loss = run_epoch(train=False, hp=hyperparams)
        printlog('Test:\t'+loss_str(test_loss))

printlog('Best Test Loss: {:.4f}'.format(best_test_loss))
