import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib import animation
from skimage.transform import resize

from . import cfg, unnormalize


######################################################################
########################### Visualizations ###########################
######################################################################


SCALE = cfg.SCALE
MACRO_SIZE = cfg.MACRO_SIZE*SCALE


def _set_figax():
    fig = plt.figure(figsize=(5,5))
    img = plt.imread(cfg.DATAPATH+'/court.png')
    img = resize(img,(SCALE*cfg.WIDTH,SCALE*cfg.LENGTH,3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    # Show just the left half-court
    ax.set_xlim([-50,550])
    ax.set_ylim([-50,550])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def _get_cmap(n_players):
    colormap = cfg.CMAP_OFFENSE
    while len(colormap) < n_players:
        colormap += cfg.DEF_COLOR
    return colormap


def display(seq, macro_intents, params=None, save_file='', colormap=cfg.CMAP_OFFENSE):
    if params['normalize']:
        seq = unnormalize(seq)

    n_players = int(len(seq[0])/2)
    colormap = _get_cmap(n_players)
    burn_in = params['burn_in']

    alpha_line = [0.5] * n_players
    alpha_dots = [0.5] * n_players

    while len(colormap) < n_players:
        colormap += cfg.DEF_COLOR

    fig, ax = _set_figax()

    for k in range(n_players):
        x = seq[:,(2*k)]
        y = seq[:,(2*k+1)]
        color = colormap[k]

        ax.plot(SCALE*x, SCALE*y, color=color, linewidth=3, alpha=alpha_line[k])
        ax.plot(SCALE*x, SCALE*y, 'o', color=color, markersize=10, alpha=alpha_dots[k])

        if macro_intents is not None:
            for t in range(len(seq)):
                if t >= burn_in:
                    m_x = int(macro_intents[t,k]/cfg.N_MACRO_Y)
                    m_y = macro_intents[t,k] - cfg.N_MACRO_Y*m_x
                    ax.add_patch(Rectangle(
                        (m_x*MACRO_SIZE, m_y*MACRO_SIZE), MACRO_SIZE, MACRO_SIZE, alpha=0.02, color=color, linewidth=2)) 

    # Starting positions
    x = seq[0,::2]
    y = seq[0,1::2]
    ax.plot(SCALE*x, SCALE*y, 'o', color='black', markersize=12)

    # Burn-ins
    if burn_in > 0:
        x = seq[:burn_in,::2]
        y = seq[:burn_in,1::2]
        ax.plot(SCALE*x, SCALE*y, color='black', linewidth=8, alpha=0.5)

    plt.tight_layout(pad=0)

    if len(save_file) > 0:
        plt.savefig(save_file+'.png')
    else:
        plt.show()


def animate(seq, macro_intents, params=None, save_file='', colormap=cfg.CMAP_OFFENSE):
    if params['normalize']:
        seq = unnormalize(seq)

    n_players = int(len(seq[0])/2)
    colormap = _get_cmap(n_players)
    burn_in = params['burn_in']
    seq_len = len(seq)

    fig, ax = _set_figax()

    trajectories = [ax.plot([],[])[0] for _ in range(n_players)]
    locations = [ax.plot([],[])[0] for _ in range(n_players)]
    burn_ins = [ax.plot([],[])[0] for _ in range(n_players)]

    macros = []
    if macro_intents is not None:
        macros = [Rectangle(xy=(0, 0), width=MACRO_SIZE, height=MACRO_SIZE, alpha=0) for k in range(macro_intents.shape[1])]                                  
    
    def init():
        for k in range(n_players):
            traj = trajectories[k]
            loc = locations[k]
            burn = burn_ins[k]
            color = colormap[k % n_players]

            traj.set_data([],[])
            traj.set_color(color)
            traj.set_linewidth(3)
            traj.set_alpha(0.7)

            loc.set_data([],[])
            loc.set_color(color)
            loc.set_marker('o')
            loc.set_markersize(12)

            burn.set_data([],[])
            burn.set_color('0.01')
            burn.set_linewidth(6)
            burn.set_alpha(0.5)

            if k < len(macros):
                m = macros[k]
                ax.add_patch(m)
                m.set_color(color)

        return trajectories+locations+burn_ins+macros

    def animate(t):
        if t >= seq_len:
            t = seq_len-1

        for p in range(n_players):
            trajectories[p].set_data(SCALE*seq[:t+1,2*p], SCALE*seq[:t+1,2*p+1])
            locations[p].set_data(SCALE*seq[t,2*p], SCALE*seq[t,2*p+1])
            burn_ins[p].set_data(SCALE*seq[:min(t, burn_in),2*p], SCALE*seq[:min(t, burn_in),2*p+1])

        # Start showing macro-intents after burn-in period
        if t >= burn_in:
            for j,m in enumerate(macros):
                m_x = int(macro_intents[t,j]/cfg.N_MACRO_Y)
                m_y = macro_intents[t,j] - cfg.N_MACRO_Y*m_x
                m.set_xy([m_x*MACRO_SIZE, m_y*MACRO_SIZE])
                m.set_alpha(0.5)

        return trajectories+locations+burn_ins+macros

    plt.tight_layout(pad=0)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=72, interval=100, blit=True)

    if len(save_file) > 0:
        anim.save(save_file+'.mp4', fps=7, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
