import os

# tf logging - don't print INFO messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

BS = 32
EP = 20  # 80
LR = 0.0003  # 0.00005

N_TRAIN = 2048  # 64  # 25000
# N_VAL = min(256, 27250 - N_TRAIN)
N_TEST = 256  # 3029

# logs, plots, visualizations, checkpoints, etc. will be saved there
LOGS_PATH = "/home/iaroslavelistratov/results"
# source code path for logging
SOURCE_PATH = "/home/iaroslavelistratov/depth/capsa"

# maps model names to their paths
base_path = "/data/capsa/depth/results/new/"
PATHS = {
    "base": base_path + "base/20220926-170005-new_callback",
    "dropout": base_path + "dropout/20220924-155901-new_callback",
    # 'vae': base_path + 'vae/20220926-104431-new_callback_sample_same-latent_20',
    "ensemble": base_path + "ensemble/20220926-103111-new_callback-4_members",
    "mve": base_path + "mve/20220926-103158-new_callback_sample_same",
}

timedelta = 3  # 3 for Iaro, 0 for ET

DATA_PATH = "/home/iaroslavelistratov/depth/data"
TRAIN_PATH = f"{DATA_PATH}/depth_train.h5"
TEST_PATH = f"{DATA_PATH}/depth_test.h5"
OOD_PATH = f"{DATA_PATH}/apolloscape_test.h5"
