import argparse
import os
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
args, _ = parser.parse_known_args()

save_dir = 'saved/{:03d}'.format(args.trial)

params = pickle.load(open(save_dir+'/params.p', 'rb'))
print(params)
