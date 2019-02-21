#########################
### Dataset constants ###
#########################

NAME = 'bball'

DATAPATH = 'datasets/{}/data'.format(NAME)
FILENAME_TRAIN = '{}_train'.format(NAME)
FILENAME_TEST = '{}_test'.format(NAME)

N_TRAIN = 107146
N_TEST = 13845
SEQUENCE_LENGTH = 50
SEQUENCE_DIMENSION = 22

############################
### Basketball constants ###
############################

LENGTH = 94
WIDTH = 50
SCALE = 10

###############################
### Data-specific constants ###
###############################

BALL = 'ball'
OFFENSE = 'offense'
DEFENSE = 'defense'

PLAYER_TYPES = [BALL, OFFENSE, DEFENSE]

COORDS = {
    BALL : { 'xy' : [0,1] },
    OFFENSE : { 'xy' : [2,3,4,5,6,7,8,9,10,11] },
    DEFENSE : { 'xy' : [12,13,14,15,16,17,18,19,20,21] }
}

for pt in PLAYER_TYPES:
    COORDS[pt]['x'] = COORDS[pt]['xy'][::2]
    COORDS[pt]['y'] = COORDS[pt]['xy'][1::2]

DEF_COLOR = 'b'
CMAP_ALL = ['orange'] + ['b']*5 + ['r']*5
CMAP_OFFENSE = ['b', 'r', 'g', 'm', 'y']

NORMALIZE = [LENGTH, WIDTH] * int(SEQUENCE_DIMENSION/2)
SHIFT = [25] * SEQUENCE_DIMENSION

##############################
### Macro-intent constants ###
##############################

SPEED_THRESHOLD = 0.5
N_MACRO_X = 9
N_MACRO_Y = 10
MACRO_SIZE = WIDTH/N_MACRO_Y
