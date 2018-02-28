import cfg
import numpy as np
import os
import pickle


DATAPATH = cfg.DATAPATH
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST
SCALE = cfg.SCALE
SEQ_LENGTH = cfg.SEQUENCE_LENGTH
X_DIM = cfg.SEQUENCE_DIMENSION
MACRO_SIZE = cfg.MACRO_SIZE
N_MACRO_X = cfg.N_MACRO_X
N_MACRO_Y = cfg.N_MACRO_Y


def bound(v, l, u):
	if v < l:
		return l
	elif v > u:
		return u
	else:
		return v


def get_macro_goal(position):
	eps = 1e-4 # hack to make calculating macro_x and macro_y cleaner
	x = bound(position[0], 0, N_MACRO_X*MACRO_SIZE-eps)
	y = bound(position[1], 0, N_MACRO_Y*MACRO_SIZE-eps)

	macro_x = int(x/MACRO_SIZE)
	macro_y = int(y/MACRO_SIZE)

	return macro_x*N_MACRO_Y + macro_y


def compute_macro_goals(track):
	velocity = track[1:,:] - track[:-1,:]
	speed = np.linalg.norm(velocity, axis=-1)
	stationary = speed < cfg.SPEED_THRESHOLD
	stationary = np.append(stationary, True) # assume last frame always stationary

	T = len(track)
	macro_goals = np.zeros(T)
	for t in reversed(range(T)):
		if t+1 == T: # assume position in last frame is always a macro goal
			macro_goals[t] = get_macro_goal(track[t])
		elif stationary[t] and not stationary[t+1]: # from stationary to moving indicated a change in macro goal
			macro_goals[t] = get_macro_goal(track[t])
		else: # otherwise, macro goal is the same
			macro_goals[t] = macro_goals[t+1]
		
	return macro_goals


train = True
N = N_TRAIN if train else N_TEST
filename = 'Xtr_role' if train else 'Xte_role'
data = np.zeros((N, SEQ_LENGTH, X_DIM))

if os.path.isfile(DATAPATH+filename+'.p'):
	data = pickle.load(open(DATAPATH+filename+'.p', 'rb'))
else:
	counter = 0
	file = open(DATAPATH+filename+'.txt')
	for line in file:
		t = counter % SEQ_LENGTH
		s = int((counter - t) / SEQ_LENGTH)
		data[s][t] = line.strip().split(' ')
		counter += 1
	pickle.dump(data, open(DATAPATH+filename+'.p', 'wb'))

macro_goals_all = np.zeros((N, SEQ_LENGTH, int(X_DIM/2)))
for i in range(N):
	for p in range(int(X_DIM/2)):
		macro_goals_all[i,:,p] = compute_macro_goals(data[i,:,2*p:2*p+2])

pickle.dump(macro_goals_all, open(DATAPATH+filename+'_macro.p', 'wb'))