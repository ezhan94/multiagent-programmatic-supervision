import argparse
import pickle
import numpy as np

import sys
sys.path.append(sys.path[0] + '/..')

from datasets.bball import cfg, unnormalize


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
args, _ = parser.parse_known_args()

offense = pickle.load(open('saved/{}/experiments/sample/samples.p'.format(args.trial), 'rb'))
offense = np.swapaxes(offense, 0, 1)
offense = unnormalize(offense)

(N, T, D) = offense.shape
n_agents = int(D/2)
print("{} samples".format(N))

speeds = np.zeros(N)
distances = np.zeros(N)
oobs = np.zeros(N)


def oob_rate(traj):
    length = traj[:,::2]
    width = traj[:,1::2]

    oob_frames = np.zeros(length.shape, dtype=bool)
    oob_frames = np.logical_or(oob_frames, length < 0)
    oob_frames = np.logical_or(oob_frames, length > cfg.LENGTH)
    oob_frames = np.logical_or(oob_frames, width < 0)
    oob_frames = np.logical_or(oob_frames, width > cfg.WIDTH)

    oob_count = np.sum(np.sum(oob_frames, axis=1) > 0) / T

    return oob_count


for i in range(N):
    traj = offense[i] # 50 x 10
    vel = traj[1:] - traj[:-1]
    vel_p = vel.reshape((-1,n_agents,2))
    speed = np.linalg.norm(vel_p, axis=-1) # 49 x 10

    speeds[i] = np.mean(speed)
    distances[i] = np.sum(speed) / n_agents
    oobs[i] = oob_rate(traj)

print("Average speed {}".format(np.mean(speeds)))
print("Average distance {}".format(np.mean(distances)))
print("Average frames out of bounds {}".format(np.mean(oobs)))
