import numpy as np
import sounddevice as sd
import argparse
import torch

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

# set some parameters
block_len_ms = 64
block_shift_ms = 50
fs_target = 16000

# calculate shift and length
# block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
# create buffer
in_buffer = np.zeros(block_len).astype('float32')
# out_buffer = np.zeros(block_len).astype('float32')

# load vad model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils



def callback(data, frames, time, status):
    # buffer and states to global
    data = torch.FloatTensor(data).squeeze()
    print(data.shape)
    speech_prob = model(data, 16000).item()
    print(speech_prob)
    return int(speech_prob)


try:
    with sd.InputStream(device=args.input_device,
                   samplerate=fs_target, blocksize=block_len,
                   dtype=np.float32, latency=args.latency,
                   channels=1, callback=callback):

        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))




