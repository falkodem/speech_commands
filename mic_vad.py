import numpy as np
import sounddevice as sd
import argparse
import torch
import tflite_runtime.interpreter as tflite

from VAD.vad_functions import init_jit_model



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


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
parser.add_argument('--latency', type=float, help='latency in seconds', default=0.2)
args = parser.parse_args(remaining)




#   ------set some parameters ------
# blocks in ms, fs in Hz
block_len_ms = 64
block_shift_ms = 8
fs_target = 16000
# create the interpreters
interpreter_1 = tflite.Interpreter(model_path='./models/model_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./models/model_2.tflite')
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
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
# number of points in block
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
# create buffer both for vad and noise suppression models
in_buffer = np.zeros(block_len).astype('float32')
out_buffer = np.zeros(block_len).astype('float32')

# load vad model
vad_model = init_jit_model('models/silero_vad.jit')

def do_smth(audio_proc):
    print(len(audio_proc), type(audio_proc))


audio_is_processed = False
thrsh = 0.5
num_of_parts = block_len_ms//block_shift_ms
current_part = 0
denoised_audio = np.zeros()

def callback(indata, frames, time, status):
    # buffer and states to global
    global in_buffer, out_buffer, states_1, states_2
    global denoised_audio, audio_is_processed, current_part

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
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    # accumulate audio fragment
    ##### outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)
    denoised_audio[] = out_buffer[:block_shift]

    # flag for word ending
    audio_is_processed = True
    # elif audio_is_processed == True:
    do_smth(denoised_audio)
    denoised_audio = []
    audio_is_processed = False

    # ---------------VAD---------------
    if current_part == num_of_parts-1:
        indata_tnsr = torch.FloatTensor(indata).squeeze()
        speech_prob = vad_model(indata_tnsr, 16000).item()

        if speech_prob > thrsh:



try:
    with sd.InputStream(device=args.input_device,
                   samplerate=fs_target, blocksize=block_shift,
                   dtype=np.float32, latency=0.5, #latency=args.latency,
                   channels=1, callback=callback):

        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

