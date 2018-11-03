import torch

# CUDA
CUDA = torch.cuda.is_available()

# Directories
CACHE_DIR = 'cache'
MODEL_DIR = 'model'
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoint'

# Decoder settings
MAX_DECODE_LEN = 15

# Corpus settings
DICT_SIZE = 4096
MIN_INPUT_LEN = 1
MAX_INPUT_LEN = 15
MIN_TARGET_LEN = 1
MAX_TARGET_LEN = 15
USE_NIL = True
TRAIN_NIL = False
