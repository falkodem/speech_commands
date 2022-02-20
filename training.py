import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os

com = SPEECHCOMMANDS("data/greetings/",download=False)
train_set = com("training")
val_set = com("validation")
test_set = com("testing")

labels = sorted(list(datapoint[2] for datapoint in train_set))
print(labels.count('duza'))