import os
import soundfile as sf
import torch
import numpy as np
from utils.vad_functions import *
from utils.utils import *
from torch.distributions import Bernoulli
import tflite_runtime.interpreter as tflite
from scipy import fft
from scipy.fftpack import dct


class VAD(torch.nn.Module):
    def __init__(self, p=0.5, mode='train_val'):
        super(VAD, self).__init__()
        self.vad_model = init_jit_model(VAD_MODEL_PATH)
        self.p = p
        self.apply_vad = Bernoulli(self.p)
        self.mode = mode

    def _get_timestamps(self, x):
        speech_timestamps = get_speech_timestamps(x,
                                                  self.vad_model,
                                                  sampling_rate=SAMPLING_RATE,
                                                  threshold=THRESHOLD)
        # check if speech_timestamps is empty, i.e. VAD did not detected any speech
        if len(speech_timestamps) != 0:
            return collect_chunks(speech_timestamps, x)
        else:
            return x

    def forward(self, x):
        x = torch.squeeze(x)
        if self.apply_vad.sample():
            x = self._get_timestamps(x)
        if self.mode == 'test':
            return x.view(1, -1)
        elif self.mode == 'train_val':
            return x.view(1, 1, -1)
        else:
            print(f"Incorrect parameter: mode = {self.mode}. Only 'test or 'train_val'")
            raise ValueError


class DTLN(torch.nn.Module):
    def __init__(self, p=0.5):
        super(DTLN, self).__init__()
        # load models
        self.interpreter_1 = tflite.Interpreter(model_path=DTLN_1_PATH)
        self.interpreter_1.allocate_tensors()
        self.interpreter_2 = tflite.Interpreter(model_path=DTLN_2_PATH)
        self.interpreter_2.allocate_tensors()

        # Get input and output tensors.
        self.input_details_1 = self.interpreter_1.get_input_details()
        self.output_details_1 = self.interpreter_1.get_output_details()

        self.input_details_2 = self.interpreter_2.get_input_details()
        self.output_details_2 = self.interpreter_2.get_output_details()
        # create states for the lstms
        self.states_1 = np.zeros(self.input_details_1[1]['shape']).astype('float32')
        self.states_2 = np.zeros(self.input_details_2[1]['shape']).astype('float32')

        self.p = p
        self.apply_vad = Bernoulli(self.p)

    def forward(self, audio):
        audio = audio.squeeze()
        if self.apply_vad.sample():
            out_file = np.zeros((len(audio)))
            # create buffer
            in_buffer = np.zeros(BLOCK_LEN).astype('float32')
            out_buffer = np.zeros(BLOCK_LEN).astype('float32')
            # calculate number of blocks
            num_blocks = (audio.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
            # iterate over the number of blocks
            for idx in range(num_blocks):
                # shift values and write to buffer
                in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
                in_buffer[-BLOCK_SHIFT:] = audio[idx * BLOCK_SHIFT:(idx * BLOCK_SHIFT) + BLOCK_SHIFT]
                # calculate fft of input block
                in_block_fft = np.fft.rfft(in_buffer)
                in_mag = np.abs(in_block_fft)
                in_phase = np.angle(in_block_fft)
                # reshape magnitude to input dimensions
                in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
                # set tensors to the first model
                self.interpreter_1.set_tensor(self.input_details_1[1]['index'], self.states_1)
                self.interpreter_1.set_tensor(self.input_details_1[0]['index'], in_mag)
                # run calculation
                self.interpreter_1.invoke()
                # get the output of the first block
                out_mask = self.interpreter_1.get_tensor(self.output_details_1[0]['index'])
                states_1 = self.interpreter_1.get_tensor(self.output_details_1[1]['index'])
                # calculate the ifft
                estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
                estimated_block = np.fft.irfft(estimated_complex)
                # reshape the time domain block
                estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
                # set tensors to the second block
                self.interpreter_2.set_tensor(self.input_details_2[1]['index'], self.states_2)
                self.interpreter_2.set_tensor(self.input_details_2[0]['index'], estimated_block)
                # run calculation
                self.interpreter_2.invoke()
                # get output tensors
                out_block = self.interpreter_2.get_tensor(self.output_details_2[0]['index'])
                states_2 = self.interpreter_2.get_tensor(self.output_details_2[1]['index'])

                # shift values and write to buffer
                out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
                out_buffer[-BLOCK_SHIFT:] = np.zeros(BLOCK_SHIFT)
                out_buffer += np.squeeze(out_block)
                # write block to output file
                out_file[idx * BLOCK_SHIFT:(idx * BLOCK_SHIFT) + BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]
            return torch.FloatTensor(out_file).view(1, -1)
        else:

            return audio

    def forward_realtime(self, audio, in_buffer, out_buffer):
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = np.squeeze(audio)
        # calculate fft of input block
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
        # set tensors to the first model
        self.interpreter_1.set_tensor(self.input_details_1[1]['index'], self.states_1)
        self.interpreter_1.set_tensor(self.input_details_1[0]['index'], in_mag)
        # run calculation
        self.interpreter_1.invoke()
        # get the output of the first block
        out_mask = self.interpreter_1.get_tensor(self.output_details_1[0]['index'])
        self.states_1 = self.interpreter_1.get_tensor(self.output_details_1[1]['index'])
        # calculate the ifft
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
        # set tensors to the second block
        self.interpreter_2.set_tensor(self.input_details_2[1]['index'], self.states_2)
        self.interpreter_2.set_tensor(self.input_details_2[0]['index'], estimated_block)
        # run calculation
        self.interpreter_2.invoke()
        # get output tensors
        out_block = self.interpreter_2.get_tensor(self.output_details_2[0]['index'])
        self.states_2 = self.interpreter_2.get_tensor(self.output_details_2[1]['index'])
        # write to buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros(BLOCK_SHIFT)
        out_buffer[-BLOCK_LEN:] = out_buffer[-BLOCK_LEN:] + np.squeeze(out_block)
        return in_buffer, out_buffer

class MFCC(torch.nn.Module):

    def __init__(self, fs, NFFT=512, num_ceps=14, normalize=False, only_mel=False):
        super(MFCC, self).__init__()
        self.fs = fs
        self.NFFT = NFFT
        self.num_ceps = num_ceps
        self.normalize = normalize
        self.only_mel = only_mel
        self.frame_size = 0.02  # 20мс
        self.frame_stride = 0.01  # шаг 10 мс
        self.frame_length = int(round(fs * self.frame_size))
        self.frame_step = int(round(fs * self.frame_stride))

    def forward(self, data):
        data = data.flatten()
        signal_length = len(data)

        num_full_frames = int(np.ceil(float(np.abs(
            signal_length - self.frame_length)) / self.frame_step))  # Число фреймов, полностью заполненных полезным сигналом

        pad_signal_length = num_full_frames * self.frame_step + self.frame_length  # длина исходного сигнала + нули в конце, чтобы стало целое число фремов
        z = np.zeros((pad_signal_length - signal_length))  # добавляем нули к последнему фрейму чтобы его заполнить
        pad_signal = np.append(data, z)  # теперь в сигнале полное число фреймов, но последний дополняется нулями
        num_frames = num_full_frames + 1  # полное число фреймов

        indices = np.tile(np.arange(0, self.frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_full_frames * self.frame_step + 1, self.frame_step),
            (self.frame_length, 1)).T  # матрица индексов: строки -фреймы, столбы - элементы фрейма
        frames = pad_signal[indices]

        # Хэмминг + Фурье
        T = 1.0 / self.fs
        w = np.hamming(self.frame_length)  # Окно Хэмминга
        frames_fourier_w = np.abs(fft.rfft(frames * w, self.NFFT))
        # Вычисляем периодограмму каждого фрейма ( спектральную мощность )
        P = ((frames_fourier_w) ** 2) / self.NFFT  # Power Spectrum
        # Вычисляем блок мел-фильтров(треугольных)
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Переводим герцы в мелы
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # равномерно распределены по мел шкале
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # мелы в герцы
        bin = np.floor((self.NFFT + 1) * hz_points / self.fs)

        fbank = np.zeros((nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(P, fbank.T).T
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        if self.normalize:
            filter_banks = filter_banks - (np.mean(filter_banks, axis=0).reshape(1, -1) + 1e-8)
            # filter_banks = filter_banks/np.max(filter_banks)
            # filter_banks = (filter_banks - np.mean(filter_banks))/np.std(filter_banks)

        if self.only_mel:
            # берем 14 коэфов
            coefs = filter_banks[1: (self.num_ceps + 1), :]
        else:
            # дискретное косинусное преобразование
            # число мел-кепстральных коэффов, остальные отбрасываем потому что они отражают быстрые изменения сигнала
            coefs = dct(filter_banks, type=2, axis=0, norm='ortho')[1: (self.num_ceps + 1), :]  # Берем 2-14

        return torch.FloatTensor(coefs).unsqueeze(0)
