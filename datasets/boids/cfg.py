#########################
### Dataset constants ###
#########################

NAME = 'boids'

DATAPATH = 'datasets/{}/data'.format(NAME)
FILENAME_TRAIN = '{}_train'.format(NAME)
FILENAME_TEST = '{}_test'.format(NAME)

N_TRAIN = 2**15
N_TEST = 2**13

SEQ_LEN = 50
N_AGENTS = 8

START = [[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
SCALE = 0.8

################################
### Model dynamics constants ###
################################

R_CLOSE = 0.2
R_LOCAL = 0.9

C_COH = 1
C_SEP = 0.1
C_ALI = 0.2
C_ORI = 1

BOUND = 2

BOOST_MIN = 0.8
BOOST_MAX = 1.4

STEP_SIZE = 0.1
