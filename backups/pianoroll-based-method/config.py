
import torch


# =========================
# Dataset
# =========================

DATA_DIR = "data"
TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"
TEST_DIR = "data/test"

NUM_TRACKS = 11
TRACK_NAMES = [
    "Drums",
    "Ensemble",
    "Pad",
    "Bass",
    "Organ",
    "Piano",
    "Guitar",
    "Brass",
    "Lead",
    "Percussive",
    "Effects"
]

MIN_PITCH = 36
MAX_PITCH = 95
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

RESOLUTION = 32

FEATURES_PER_VALUE = 3
NUM_OUTPUTS = NUM_TRACKS * NUM_PITCHES
FEATURES_PER_TOKEN = NUM_OUTPUTS * FEATURES_PER_VALUE


# =========================
# Transformer
# =========================

D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
MAX_SEQ_LEN = 512
DROPOUT = 0.1


# =========================
# Training
# =========================

BATCH_SIZE = 8
NUM_EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

NUM_WORKERS = 4
PIN_MEMORY = True

USE_AMP = True

PRINT_INTERVAL = 1
SAVE_INTERVAL = 5

# =========================
# Loss weights
# =========================

PITCH_LOSS_WEIGHT = 1.0
VELOCITY_LOSS_WEIGHT = 1.0
DURATION_LOSS_WEIGHT = 1.0


# =========================
# Checkpoints
# =========================

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "checkpoints/best_model.pth"
LAST_MODEL_PATH = "checkpoints/last_model.pth"

SAVE_EVERY_EPOCH = True


# =========================
# Generation
# =========================

GENERATION_LENGTH = 2048
GENERATION_TEMPERATURE = 1.0
PITCH_THRESHOLD = 0.5


# =========================
# Device
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
