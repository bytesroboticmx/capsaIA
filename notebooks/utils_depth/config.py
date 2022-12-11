import os

# tf logging - don't print INFO messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

BS = 32
EP = 20
LR = 0.0003

N_TRAIN = 8192
N_TEST = 256

DATA_PATH = "/home/iaroslavelistratov/depth/data"
TRAIN_PATH = f"{DATA_PATH}/depth_train.h5"
TEST_PATH = f"{DATA_PATH}/depth_test.h5"
OOD_PATH = f"{DATA_PATH}/apolloscape_test.h5"
