import json
import os

from tqdm import tqdm
import torchaudio

from utils.data_loaders import LoaderCreator
from utils.utils import *

aug_loader = LoaderCreator.augmentation_loader(path=DATA_DIR, prob=0.25)

try:
    with open('./logs/aug_files.json', 'r') as f:
        indexes = json.load(f)
        print('Adding files...')
except FileNotFoundError:
    indexes = {}
    print('Rewriting files...')

replication_rate = 20
for epoch in range(replication_rate):
    for data, label in tqdm(aug_loader):
        if indexes.get(label[0]) is None:
            indexes[label[0]] = 0
            os.mkdir(f'./data/augmented/{label[0]}')
        else:
            indexes[label[0]]+=1

        torchaudio.save(f'./data/augmented/{label[0]}/{label[0]}_{indexes[label[0]]}.wav',
                        data.squeeze(1),
                        SAMPLING_RATE)

with open('./logs/aug_files.json', 'w') as f:
    json.dump(indexes, f)
