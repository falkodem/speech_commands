import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio import transforms as sound_transforms
from torch_audiomentations import AddBackgroundNoise, AddColoredNoise, \
     HighPassFilter, LowPassFilter, PolarityInversion, Shift
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from pathlib import Path

from utils.preprocessing import VAD_torch, DTLN_torch
from utils.utils import *


class FirstModelDataset(Dataset):
    def __init__(self, files, mode, prob):
        """
        :param files: list of audio files
        :param mode: generate train, val or test Dataset
        :param prob: probability of applying augmentation
        """
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.prob = prob

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        sr, audio = wavfile.read(file)
        if sr != SAMPLING_RATE:
            print(f"Sample rate of audio is {sr} Hz. Only supports {SAMPLING_RATE} Hz")
            raise ValueError
        else:
            return audio

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = torch.nn.Sequential(
            VAD_torch(p=1, mode='test'),
            DTLN_torch(p=1),
            sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            transforms.Resize((SIZE_Y, SIZE_X))  # ИЛИ ПЭДДИНГ???
        )
        data_transforms = torch.nn.Sequential(
            AddBackgroundNoise(BACKGROUND_NOISE_PATH, min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                               p=self.prob, sample_rate=SAMPLING_RATE),
            AddColoredNoise(min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                            p=self.prob, sample_rate=SAMPLING_RATE),
            DTLN_torch(p=self.prob),
            VAD_torch(p=self.prob),
            sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            transforms.Resize((SIZE_Y, SIZE_X))  # ИЛИ ПЭДДИНГ???
            # FrequencyMasking
            # TimeMasking
            # TimeStretch
            # SlidingWindowCmn???

            # HighPassFilter(),
            # LowPassFilter(),
            # PolarityInversion(),
            # Shift(),
            # Сначала нужно нормализовать сигнал от -1 до 1
            # шум белый, розовый разной интенсивности - нормализовать
            # шум с борта - нормализовать
            # очищение от шума
            # LPF HPS BandPass
            # а вообще вот https://github.com/iver56/audiomentations и GPU https://github.com/asteroid-team/torch-audiomentations
            # FrequencyMask - есть в торчаудио
            # Reverb
            # растянуть/сжать
            # кроп начала или конца
        )
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        if self.mode == 'train':
            x = data_transforms(x).squeeze()
        else:
            x = transform(x).squeeze()
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, audio):
        x = torch.FloatTensor(audio)
        return ((x - x.mean()) / x.std()).view(1, 1, -1)


TRAIN_DIR = Path('SpeechCommands/')
TEST_DIR = Path('SpeechCommands/')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.wav')))
test_files = sorted(list(TEST_DIR.rglob('*.wav')))
train_val_labels = [path.parent.name for path in train_val_files]

print(len(train_val_files))
train_dataset = FirstModelDataset(train_val_files, mode='train', prob=0.5)
test_dataset = FirstModelDataset(train_val_files, mode='test', prob=0.5)

print('Train', train_dataset.__len__())
print('test', test_dataset.__len__())

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

print('----------------')
for inputs, labels in loader:
    print(inputs.shape)
    print(labels)
    a = inputs.squeeze()
    break
# wavfile.write('out.wav',SAMPLING_RATE, a.numpy())
f1, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(a[0], annot=False, vmin=0, fmt=".1f")
ax.invert_yaxis()
plt.show()
