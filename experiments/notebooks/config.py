import os

# tf logging - don't print INFO messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

BS = 32
EP = 20
LR = 0.0003

N_TRAIN = 8192
N_TEST = 256

# maps model names to their paths
base_path = "/home/iaroslavelistratov/results/"
PATHS = {
    "base": base_path + "base/20221208-180920",
    "dropout": base_path + "dropout/20221208-192959",
    # 'vae': base_path + 'vae/20220926-104431',
    "ensemble": base_path + "ensemble/20221208-215136-3_members",
    "mve": base_path + "mve/20221208-193015",
}

timedelta = 3  # 3 for Iaro, 0 for ET

DATA_PATH = "/home/iaroslavelistratov/depth/data"
TRAIN_PATH = f"{DATA_PATH}/depth_train.h5"
TEST_PATH = f"{DATA_PATH}/depth_test.h5"
OOD_PATH = f"{DATA_PATH}/apolloscape_test.h5"
