import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio import transforms as sound_transforms
from torch_audiomentations import AddBackgroundNoise, AddColoredNoise, \
     HighPassFilter, LowPassFilter, PolarityInversion, Shift
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from pathlib import Path

from utils.preprocessing import VAD, DTLN, MFCC
from utils.utils import *


class SpeechDataset(Dataset):
    def __init__(self, files, labels, mode, prob, to_disc=False, from_disc=False):
        """
        :param files: list of audio files
        :param labels: list of labels
        :param mode: generate train, val or test Dataset
        :param prob: probability of applying augmentation (not used if from_disc=True)
        :to_disc: True if you wand to augment and write audio on disc (using <augmentation_loader> method),
         False if you want to augment on the fly
        :from_disc: True if you want to load augmented dataset from disc, without applying any augmentation
        """
        super().__init__()
        self.wake_up_word = WAKE_UP_WORD
        self.files = files
        self.labels = labels
        self.mode = mode
        self.prob = prob
        self.to_disc = to_disc
        self.from_disc = from_disc

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        # обработка теста и валидации
        self.transform = torch.nn.Sequential(
            DTLN(p=1),
            VAD(p=1, mode='test'),
            # sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            MFCC(fs=SAMPLING_RATE, num_ceps=NUM_CEPS, normalize=True, only_mel=False),
            transforms.Resize((SIZE_Y, SIZE_X))  # ИЛИ ПЭДДИНГ???
        )
        # обработка трейна
        self.aug_transforms = torch.nn.Sequential(
            AddBackgroundNoise(BACKGROUND_NOISE_PATH, min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                               p=self.prob, sample_rate=SAMPLING_RATE),
            AddColoredNoise(min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                            p=self.prob, sample_rate=SAMPLING_RATE),
            DTLN(p=self.prob),
            VAD(p=self.prob),
            # sound_transforms.Spectrogram(n_fft=200),
            # sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            MFCC(fs=SAMPLING_RATE, num_ceps=NUM_CEPS, normalize=True, only_mel=False),
            transforms.Resize((SIZE_Y, SIZE_X)),  # ИЛИ ПЭДДИНГ???
            sound_transforms.FrequencyMasking(freq_mask_param=3),
            sound_transforms.TimeMasking(16),
            # sound_transforms.TimeStretch(),
            # sound_transforms.SlidingWindowCmn()
            # HighPassFilter(p=self.prob)
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

        self.aug_transforms_no_mfcc = torch.nn.Sequential(
            AddBackgroundNoise(BACKGROUND_NOISE_PATH, min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                               p=self.prob, sample_rate=SAMPLING_RATE),
            AddColoredNoise(min_snr_in_db=NOISE_MIN_SNR, max_snr_in_db=NOISE_MAX_SNR,
                            p=self.prob, sample_rate=SAMPLING_RATE),
            DTLN(p=self.prob),
            VAD(p=self.prob)
        )

        self.aug_transforms_no_preproc = torch.nn.Sequential(
            # sound_transforms.Spectrogram(n_fft=100,normalized=False),
            MFCC(fs=SAMPLING_RATE, num_ceps=NUM_CEPS, normalize=True, only_mel=False),
            # sound_transforms.MFCC(sample_rate=SAMPLING_RATE, n_mfcc=N_MFCC),
            transforms.Resize((SIZE_Y, SIZE_X)),  # ИЛИ ПЭДДИНГ???
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
        if self.to_disc:
            x = self.aug_transforms_no_mfcc(x).squeeze()
        elif self.from_disc:
            x = self.aug_transforms_no_preproc(x).squeeze()

            # x = x + 1e-10

            # print(self.mode)
            # print('idx:',index)
            # print('label:',self.labels[index])
            # print(self.files[index])
            # print('-'*10)

            # f1, ax = plt.subplots(figsize=(15, 6))
            # sns.heatmap(x, annot=False, vmin=0, fmt=".1f")
            # ax.invert_yaxis()
            # plt.show()
        else:
            if self.mode == 'train':
                x = self.aug_transforms(x).squeeze()
            else:
                x = self.transform(x).squeeze()

        y = self.labels[index]
        return torch.unsqueeze(x, 0), y

    def _prepare_sample(self, audio):
        x = torch.FloatTensor(audio)
        return ((x - x.mean()) / x.std()).view(1, 1, -1)


class LoaderCreator:
    def __init__(self,
                 path,
                 model_type='wake_up',
                 validation=False,
                 val_size=0.15,
                 test_size=0.15,
                 batch_size=1024,
                 prob=0.5,
                 from_disc=False,
                 seed=42):
        """
        :model_type: 'wake_up' model or command 'detector' model
        :param path: directory with audio
        :param validation: create validation dataset or not
        :param val_size: val dataset fraction
        :param test_size: test dataset fraction
        :batch_size: batch size
        :prob: probability of applying augmentation transform (only for train)
        :from_disc: True if you want to load augmented dataset from disc, without applying any augmentation
        :seed: random seed for train-test split
        """
        self.wake_up_word = WAKE_UP_WORD
        self.validation = validation
        self.model_type = model_type
        self.files = sorted(list(Path(path).rglob('*.wav')))
        self.from_disc = from_disc
        self.seed = seed

        if self.model_type == 'detector':
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform([pth.parent.name for pth in self.files])
            with open('../data/label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)
        elif self.model_type == 'wake_up':
            self.labels = [1 if pth.parent.name == self.wake_up_word else 0 for pth in self.files]
        else:
            print(f"{self.model_type} is not correct; correct model_types: 'detector' or 'wake_up'")
            raise NameError

        if self.validation:
            self.split_size1 = val_size + test_size
            self.split_size2 = test_size/(val_size+test_size)
        else:
            self.split_size1 = test_size
            self.split_size2 = 0

        self.batch_size = batch_size
        self.prob = prob

    def get_loaders(self):
        train_files, test_files, train_labels, test_labels = train_test_split(self.files,
                                                                              self.labels,
                                                                              test_size=self.split_size1,
                                                                              stratify=self.labels,
                                                                              shuffle=True,
                                                                              random_state=self.seed)

        if self.validation:
            val_files, test_files, val_labels, test_labels = train_test_split(test_files,
                                                                              test_labels,
                                                                              test_size=self.split_size2,
                                                                              stratify=test_labels,
                                                                              shuffle=True,
                                                                              random_state=self.seed)
        # print('**'*20,test_files[:6], test_labels[:6])
        train_dataset = SpeechDataset(files=train_files, labels=train_labels, mode='train', prob=self.prob,
                                      from_disc=self.from_disc)
        # print('%%' * 20, train_dataset.files[:6], train_dataset.labels[:6])
        test_dataset = SpeechDataset(files=test_files, labels=test_labels, mode='test', prob=1,
                                     from_disc=self.from_disc)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.validation:
            val_dataset = SpeechDataset(files=val_files, labels=val_labels, mode='val', prob=1,
                                        from_disc=self.from_disc)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            val_loader = None

        return train_loader, val_loader, test_loader

    @staticmethod
    def augmentation_loader(path, prob):
        """
        This method id used for creating augmented audiofiles for writing on the disc.
        The reason for it is that at this moment VAD и DTLN are the bottleneck of "on the fly" augmentation,
        so training process increases in 20-50 times compared to not using these augmentation methods.
        """
        files = sorted(list(Path(path).rglob('*.wav')))
        labels = [path.parent.name for path in files]
        aug_dataset = SpeechDataset(files=files, labels=labels, mode='train', prob=prob, to_disc=True)
        dataset_loader = DataLoader(aug_dataset, batch_size=1, shuffle=False)

        return dataset_loader
