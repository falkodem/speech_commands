import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import roc_curve

SEED = 173
WAKE_UP_WORD = 'evridika'
VAD_MODEL_PATH = 'models/files/silero_vad.jit'
DTLN_1_PATH = 'models/files/model_1.tflite'
DTLN_2_PATH = 'models/files/model_2.tflite'
BACKGROUND_NOISE_PATH = 'data/noise/'

SAMPLING_RATE = 16000
SIZE_X = 128
SIZE_Y = 32
NUM_CEPS = 32

# Infer VAD and DTLN params
BLOCK_LEN_MS = 32  # block len in ms
BLOCK_SHIFT_MS = 8  # block shift in ms
BLOCK_LEN_MS_VAD = 128
BLOCK_SHIFT = int(np.round(SAMPLING_RATE * (BLOCK_SHIFT_MS / 1000)))
# number of points in block
BLOCK_LEN = int(np.round(SAMPLING_RATE * (BLOCK_LEN_MS / 1000)))
NUM_OF_PARTS = BLOCK_LEN_MS_VAD//BLOCK_SHIFT_MS
BLOCK_LEN_VAD = int(np.round(SAMPLING_RATE * (BLOCK_LEN_MS_VAD / 1000)))
VAD_THRESHOLD = 0.1  # VAD classification threshold
LATENCY = 0.7

# Training params
BATCH_SIZE = 196
N_CHANNEL_WAKE_UP = 8
N_EPOCH = 60
LOG_INTERVAL = 20
VAL_SIZE = 0.1
TEST_SIZE = 0.1
PROB = 0.25
N_MFCC = 32
NOISE_MIN_SNR = -10
NOISE_MAX_SNR = 15

# other
DATA_MODES = ['train', 'val', 'test']
N_BOOTSTRP = 10000
SPEC_THRSH = 0.995
WAKE_UP_THRSH = 0.67


DATA_DIR = Path('data/commands/')
DATA_DIR_AUGMENTED = Path('data/augmented')
SAVE_MODEL_DIR = 'models/files/'

# Expert system
CMD_ACCEPT_THRSH = 0.4


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def accuracy(labels, preds):
    return (preds.argmax(dim=1) == labels).float().mean().data.cpu()


def rec_at_spec(labels, preds, spec_thrsh=0.999):
    fpr, rec, thrsh = roc_curve(labels, preds)
    spec = 1 - fpr
    best_idx = np.argmax(np.where(spec >= spec_thrsh))
    return spec[best_idx], rec[best_idx], thrsh[best_idx]


def bootstrap(labels, preds, n_bootstrps, thrsh):
    TP = []
    FP = []
    TN = []
    FN = []
    sample_size = len(labels)
    for iter_bootstrp in range(n_bootstrps):
        idxs = np.random.choice(sample_size)
        preds_sample = preds[idxs]
        labels_sample = labels[idxs].astype('bool')
        TP.append(np.sum(labels_sample == (preds_sample >= thrsh)))
        FP.append(np.sum(~labels_sample == (preds_sample >= thrsh)))
        TN.append(np.sum(~labels_sample == (preds_sample < thrsh)))
        FN.append(np.sum(labels_sample == (preds_sample < thrsh)))

    return np.array(TP), np.array(FP), np.array(TN), np.array(FN)
