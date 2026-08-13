from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Directories
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# Dataset
DATASET_NAME = "dair-ai/emotion"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


# Emotion labels
LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

NUM_CLASSES = len(LABEL_NAMES)


# Reproducibility
RANDOM_SEED = 42


# Model
VOCAB_SIZE = 20_000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3


# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1