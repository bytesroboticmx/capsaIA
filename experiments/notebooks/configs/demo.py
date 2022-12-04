### used for demo


import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BS = 32
EP = 70 # 20
LR = 0.0003 # 0.00005

N_TRAIN = 25000
N_VAL = 27250 - N_TRAIN

# source code path for logging
SOURCE_PATH = '/home/iaroslavelistratov/capsa'
# logs, plots, visualizations, checkpoints, etc. will be saved there
LOGS_PATH = '/home/iaroslavelistratov/results'
# load model from
BASE_PATH = '/home/iaroslavelistratov/results/base/20220926-170005-new_callback'
ALEATORIC_PATH = '/home/iaroslavelistratov/results/mve/20220926-103158-new_callback_sample_same'
EPISTEMIC_PATH = '/home/iaroslavelistratov/results/ensemble/20220926-103111-new_callback-4_members'


timedelta = 3 # 3 for Iaro, 0 for ET