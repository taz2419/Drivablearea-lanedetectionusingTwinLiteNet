import torch
from torch import nn
from DataSet import myImageFloder

# Dataset paths
DATASET_DIR = './data/bdd100k'

# Training constants
NUM_CLASSES_DA = 2  # Drivable area classes
NUM_CLASSES_LL = 2  # Lane line classes

# Image dimensions
IMG_HEIGHT = 368
IMG_WIDTH = 640

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"