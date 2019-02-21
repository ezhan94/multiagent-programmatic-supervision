import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from . import cfg


######################################################################
########################### Visualizations ###########################
######################################################################


def _set_figax():
    boundary = cfg.BOUND

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    min_val = -boundary-0.5
    max_val = boundary+0.5
    ax.set_xlim([min_val,max_val])
    ax.set_ylim([min_val,max_val])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    circle = Circle((0, 0), radius=boundary+0.3, fill=False, linestyle='dashed', alpha=0.5)
    ax.add_patch(circle)

    return fig, ax


def display(seq, macro_intents, params=None, save_file=''):
    n_players = int(len(seq[0])/2)

    color, marker = 'b', 'o'
    if macro_intents is not None and macro_intents[0] == 0:
        color, marker = 'r', 'v'

    fig, ax = _set_figax()

    for k in range(n_players):
        x = seq[:,(2*k)]
        y = seq[:,(2*k+1)]

        ax.plot(x, y, color=color, linewidth=3, alpha=0.3)

    # End positions
    x = seq[-1,::2]
    y = seq[-1,1::2]
    ax.plot(x, y, marker, color=color, markersize=12, markeredgewidth=2, alpha=0.9)

    plt.tight_layout(pad=0)

    if len(save_file) > 0:
        plt.savefig(save_file+'.png')
    else:
        plt.show()


def animate(seq, macro_intents, params=None, save_file=''):
    n_players = int(len(seq[0])/2)
    seq_len = len(seq)

    fig, ax = _set_figax()

    trajectories = [ax.plot([],[])[0] for _ in range(n_players)]
    locations = [ax.plot([],[])[0] for _ in range(n_players)]

    def init():
        for k in range(n_players):
            traj = trajectories[k]
            loc = locations[k]

            color, marker = 'b', 'o'
            if macro_intents is not None and macro_intents[0] == 0:
                color, marker = 'r', 'X'

            traj.set_data([],[])
            traj.set_color(color)
            traj.set_linewidth(1)
            traj.set_alpha(0.1)

            loc.set_data([],[])
            loc.set_color(color)
            loc.set_marker(marker)
            loc.set_markersize(10)

        return trajectories+locations

    def animate(t):
        if t >= seq_len:
            t = seq_len-1

        for p in range(n_players):
            trajectories[p].set_data(seq[:t+1,2*p], seq[:t+1,2*p+1])
            locations[p].set_data(seq[t,2*p], seq[t,2*p+1])

        return trajectories+locations

    plt.tight_layout(pad=0)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=72, interval=100, blit=True)

    if len(save_file) > 0:
        anim.save(save_file+'.mp4', fps=7, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
        