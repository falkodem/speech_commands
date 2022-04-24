import os
import soundfile as sf
import torch
import numpy as np
from utils.vad_functions import *
from utils.utils import *
from torch.distributions import Bernoulli
import tflite_runtime.interpreter as tflite


class VAD_torch(torch.nn.Module):
    def __init__(self, p=0.5, mode='train_val'):
        super(VAD_torch, self).__init__()
        self.vad_model = init_jit_model(VAD_MODEL_PATH)
        self.p = p
        self.apply_vad = Bernoulli(self.p)
        self.mode = mode

    def _get_timestamps(self, x):
        speech_timestamps = get_speech_timestamps(x, self.vad_model, sampling_rate=SAMPLING_RATE)
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


class DTLN_torch(torch.nn.Module):
    def __init__(self, p=0.5):
        super(DTLN_torch, self).__init__()
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
