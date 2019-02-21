import cfg
import os
import numpy as np

from boid import Boid, BoidList


# How often we randomly change the velocity
# 0 means never (i.e. no additional stochasticity)
stochastic_interval = 10 

n_agents = cfg.N_AGENTS
seq_len = cfg.SEQ_LEN
datapath = cfg.DATAPATH
macro_path = os.path.join(cfg.DATAPATH, "macro_intents")

for i in range(2):
    # Create filename
    filename = cfg.FILENAME_TEST if i == 0 else cfg.FILENAME_TRAIN

    # Create save destination
    if not os.path.exists(datapath):
        os.makedirs(datapath)
        os.makedirs(macro_path)
    fullpath = os.path.join(cfg.DATAPATH, filename)

    # Create data and label arrays
    n = cfg.N_TEST if i == 0 else cfg.N_TRAIN
    data = np.zeros((n, seq_len, 2*n_agents))
    labels = np.zeros((n, seq_len, 1))

    for s in range(n):
        # Initialize behaviors
        start = cfg.SCALE*np.array(cfg.START, dtype='f')
        behavior = np.random.randint(2)

        # Initialize Boids
        boids = BoidList()
        for j in range(n_agents):
            newBoid = Boid(pos=start[j], vel=np.random.randn(2), attract=(behavior==1))
            boids.add(newBoid)

        # Set initial conditions
        data[s,0] = np.reshape(start, -1)
        labels[s,0] = behavior

        # Generate trajectories
        for t in range(1, seq_len):
            if stochastic_interval > 0 and t % stochastic_interval == 0:
                boids.sample_boost()
            boids.update_velocities()
            boids.step()

            data[s,t] = np.reshape(boids.get_attributes('pos'), -1)
            labels[s,t] = behavior # later on, can also change behaviors during rollout

    # Save data and labels
    np.savez(fullpath+'.npz', data=data)
    np.savez(os.path.join(macro_path, "{}_friendly.npz".format(filename)), data=labels)
