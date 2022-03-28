import numpy as np
import sounddevice as sd
import argparse
import torch
import tflite_runtime.interpreter as tflite
import torchaudio.backend.soundfile_backend
from utils.vad_functions import init_jit_model
from utils.utils import *


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument('--latency', type=float, help='latency in seconds', default=0.2)
args = parser.parse_args(remaining)




#   ------set some parameters ------
# blocks in ms, fs in Hz
block_len_ms = 32
block_shift_ms = 8
num_parts = block_len_ms//block_shift_ms
# create the interpreters
interpreter_1 = tflite.Interpreter(model_path=DTLN_1_PATH)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path=DTLN_2_PATH)
interpreter_2.allocate_tensors()
# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
# create states for the lstms
states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')
# calculate block shift for noise suppression model
block_shift = int(np.round(SAMPLING_RATE * (block_shift_ms / 1000)))
# number of points in block
block_len = int(np.round(SAMPLING_RATE * (block_len_ms / 1000)))
# create buffer both for vad and noise suppression models
in_buffer = np.zeros(block_len).astype('float32')
out_buffer = np.zeros(block_len).astype('float32')

# load vad model
vad_model = init_jit_model(VAD_MODEL_PATH)

def do_smth(audio_proc):
    a = torch.FloatTensor(audio_proc).reshape(1, -1)
    torchaudio.backend.soundfile_backend.save('test.wav', a, SAMPLING_RATE)


audio_is_processed = False
thrsh = 0.1
num_of_parts = block_len_ms//block_shift_ms
current_part = 0
denoised_speech = []
denoised_audio = out_buffer.copy()
i=0

def callback(indata, frames, time, status):
    # buffer and states to global
    global in_buffer, out_buffer, states_1, states_2
    global denoised_speech,denoised_audio, audio_is_processed, current_part,i

    if status:
        print(status)

    # ---------------DTLN--------------
    # write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    # run calculation
    interpreter_1.invoke()
    # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros(block_shift)
    out_buffer[-block_len:] = out_buffer[-block_len:] + np.squeeze(out_block)

    denoised_audio[:-block_shift] = denoised_audio[block_shift:]
    denoised_audio[-block_shift:] = np.zeros(block_shift)
    denoised_audio[-block_shift:] = out_buffer[:block_shift]

    # ---------------VAD---------------
    # Добавить экспоненциальное сглаживание вероятности речи, чтобы резко не обрывалось и не было провалов
    if current_part == num_of_parts-1:
        current_part = -1
        vad_indata_tnsr = torch.FloatTensor(denoised_audio).squeeze()
        speech_prob = vad_model(vad_indata_tnsr, SAMPLING_RATE).item()
        print(speech_prob)
        i+=1
        if speech_prob > thrsh:
            audio_is_processed = True
            denoised_speech.extend(denoised_audio)
        elif audio_is_processed == True:
            audio_is_processed = False
            do_smth(denoised_speech)
            denoised_speech = []

    current_part+=1


try:
    with sd.InputStream(device=args.input_device,
                   samplerate=SAMPLING_RATE, blocksize=block_shift,
                   dtype=np.float32, latency=0.5,
                   channels=1, callback=callback):

        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

