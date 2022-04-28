import numpy as np
from pathlib import Path
import torch

WAKE_UP_WORD = 'evridika'
VAD_MODEL_PATH = 'models/files/silero_vad.jit'
DTLN_1_PATH = 'models/files/model_1.tflite'
DTLN_2_PATH = 'models/files/model_2.tflite'
BACKGROUND_NOISE_PATH = 'data/noise/'

SAMPLING_RATE = 16000
SIZE_X = 256
SIZE_Y = 64
BLOCK_LEN_MS = 32  # block len in ms
BLOCK_SHIFT_MS = 8  # block shift in ms

BLOCK_LEN = int(np.round(SAMPLING_RATE * (BLOCK_LEN_MS / 1000)))  # block len in samples
BLOCK_SHIFT = int(np.round(SAMPLING_RATE * (BLOCK_SHIFT_MS / 1000)))  # block shift in samples

THRESHOLD = 0.1  # classification threshold
DATA_MODES = ['train', 'val', 'test']

N_MFCC = 32
NOISE_MIN_SNR = 0
NOISE_MAX_SNR = 15

DATA_DIR = Path('data/commands/')
DATA_DIR_AUGMENTED = Path('data/augmented')
SAVE_MODEL_DIR = 'models/files/'


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def accuracy(labels, preds):
    return (preds.argmax(dim=1) == labels).float().mean().data.cpu()

