DATAPATH = 'bball_data/data/'

############################
### basketball constants ###
############################

LENGTH = 94
WIDTH = 50

###############################
### data-specific constants ###
###############################

TRAIN = 'train'
TEST = 'test'
BALL = 'ball'
OFFENSE = 'offense'
DEFENSE = 'defense'

MODELS_DIRECTORY = 'saved-models-bball'

N_TRAIN = 107146
N_TEST = 13845
SCALE = 10
SEQUENCE_LENGTH = 50
SEQUENCE_DIMENSION = 22

PLAYER_TYPES = [BALL, OFFENSE, DEFENSE]

COORDS = {
	BALL : { 'xy' : [0,1] },
	OFFENSE : { 'xy' : [2,3,4,5,6,7,8,9,10,11] },
	DEFENSE : { 'xy' : [12,13,14,15,16,17,18,19,20,21] }
}

for pt in PLAYER_TYPES:
	COORDS[pt]['x'] = COORDS[pt]['xy'][::2]
	COORDS[pt]['y'] = COORDS[pt]['xy'][1::2]

CMAP_ALL = ['orange'] + ['b']*5 + ['r']*5
CMAP_OFFENSE = ['b', 'r', 'g', 'm', 'y']

NORMALIZE = [LENGTH, WIDTH] * int(SEQUENCE_DIMENSION/2)
SHIFT = [25] * SEQUENCE_DIMENSION

############################
### macro goal constants ###
############################

SPEED_THRESHOLD = 0.5
N_MACRO_X = 9
N_MACRO_Y = 10
MACRO_SIZE = WIDTH/N_MACRO_Y