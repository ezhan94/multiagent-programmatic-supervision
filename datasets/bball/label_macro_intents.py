import argparse
import numpy as np
import os

from . import cfg


MACRO_SIZE = cfg.MACRO_SIZE
N_MACRO_X = cfg.N_MACRO_X
N_MACRO_Y = cfg.N_MACRO_Y


def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


def get_macro_intent(position):
    """Computes the macro-intent index."""
    eps = 1e-4 # hack to make calculating macro_x and macro_y cleaner
    x = bound(position[0], 0, N_MACRO_X*MACRO_SIZE-eps)
    y = bound(position[1], 0, N_MACRO_Y*MACRO_SIZE-eps)

    macro_x = int(x/MACRO_SIZE)
    macro_y = int(y/MACRO_SIZE)

    return macro_x*N_MACRO_Y + macro_y


def compute_macro_intents_stationary(track):
    """Computes macro-intents as next stationary points in the trajectory."""
    velocity = track[1:,:] - track[:-1,:]
    speed = np.linalg.norm(velocity, axis=-1)
    stationary = speed < cfg.SPEED_THRESHOLD
    stationary = np.append(stationary, True) # assume last frame always stationary

    T = len(track)
    macro_intents = np.zeros(T)
    for t in reversed(range(T)):
        if t+1 == T: # assume position in last frame is always a macro intent
            macro_intents[t] = get_macro_intent(track[t])
        elif stationary[t] and not stationary[t+1]: # from stationary to moving indicated a change in macro intent
            macro_intents[t] = get_macro_intent(track[t])
        else: # otherwise, macro intent is the same
            macro_intents[t] = macro_intents[t+1]
    
    return macro_intents


def compute_macro_intents_fixed(track, window=1):
    """Computes macro-intents using the position at the end of each window."""
    T = len(track)
    macro_intents = np.zeros(T)
    for t in reversed(range(T)):
        if (t+1) % window == 0 or (t+1) == T:
            macro_intents[t] = get_macro_intent(track[t])
        else:
            macro_intents[t] = macro_intents[t+1]
    
    return macro_intents


def label_macro_intents(window_size=0):
    """Computes and saves labeling functions for basketball.
    Args:
        window_size (int): If positive, will label macro-intents every window_size timesteps.
                            Otherwise, will label stationary positions as macro-intents.
    """

    for i in range(2):
        train = (i == 0)

        N = cfg.N_TRAIN if train else cfg.N_TEST
        N_AGENTS = int(cfg.SEQUENCE_DIMENSION/2)
        
        # Load data
        filename = cfg.FILENAME_TRAIN if train else cfg.FILENAME_TEST
        fullpath = os.path.join(cfg.DATAPATH, filename)
        data = np.load("{}.npz".format(fullpath))['data']
        assert data.shape == (N, cfg.SEQUENCE_LENGTH, cfg.SEQUENCE_DIMENSION)

        # Compute macro-intents
        macro_intents_all = np.zeros((N, data.shape[1], N_AGENTS))

        for i in range(N):
            for k in range(N_AGENTS):
                if window_size > 0:
                    macro_intents_all[i,:,k] = compute_macro_intents_fixed(data[i,:,2*k:2*k+2], window=window_size)
                else:
                    macro_intents_all[i,:,k] = compute_macro_intents_stationary(data[i,:,2*k:2*k+2])

        # Save macro-intents
        save_dir = os.path.join(cfg.DATAPATH, "macro_intents")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if window_size > 0:
            save_path = os.path.join(save_dir, "{}_macro_window{}.npz".format(filename, window_size))
        else:
            save_path = os.path.join(save_dir, "{}_macro_stationary.npz".format(filename))

        np.savez(save_path, data=macro_intents_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ws', '--window_size', type=int,
                        required=False, default=0,
                        help='window size for labeling function')
    args = parser.parse_args()

    label_macro_intents(window_size=args.window_size)
