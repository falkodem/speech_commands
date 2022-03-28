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
    def __init__(self, files, mode, prob, model_type):
        """
        :param files: list of audio files
        :param mode: generate train, val or test Dataset
        :param prob: probability of applying augmentation
        :model_type: 'wake_up' model or  command 'detector' model
        """
        super().__init__()
        self.wake_up_word = WAKE_UP_WORD
        self.files = sorted(files)
        self.mode = mode
        self.prob = prob
        self.model_type = model_type

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        if self.model_type == 'detector':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)
            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)
        elif self.model_type == 'wake_up':
            self.labels = [1 if path.parent.name==self.wake_up_word else 0 for path in self.files]
        else:
            print(f"{self.model_type} is not correct; correct model_types: 'detector' or 'wake_up'")
            raise NameError

        # для преобразования изображений в тензоры PyTorch и нормализации входа
        self.transform = torch.nn.Sequential(
            VAD_torch(p=1, mode='test'),
            DTLN_torch(p=1),
            sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            transforms.Resize((SIZE_Y, SIZE_X))  # ИЛИ ПЭДДИНГ???
        )
        self.data_transforms = torch.nn.Sequential(
            AddBackgroundNoise(BACKGROUND_NOISE_PATH, min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                               p=self.prob, sample_rate=SAMPLING_RATE),
            AddColoredNoise(min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                            p=self.prob, sample_rate=SAMPLING_RATE),
            DTLN_torch(p=self.prob),
            VAD_torch(p=self.prob),
            sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            transforms.Resize((SIZE_Y, SIZE_X)),  # ИЛИ ПЭДДИНГ???
            # sound_transforms.FrequencyMasking(),
            # sound_transforms.TimeMasking(),
            # sound_transforms.TimeStretch(),
            # sound_transforms.SlidingWindowCmn()
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
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        if self.mode == 'train':
            x = self.data_transforms(x).squeeze()
        else:
            x = self.transform(x).squeeze()

        if self.model_type == 'detector':
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
        else:
            y = self.labels[index]
        return x, y

    def _prepare_sample(self, audio):
        x = torch.FloatTensor(audio)
        return ((x - x.mean()) / x.std()).view(1, 1, -1)

