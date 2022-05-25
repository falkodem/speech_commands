import pickle
import sounddevice as sd
import argparse
from utils.vad_functions import init_jit_model
from utils.utils import *
from utils.preprocessing import DTLN
from utils.system_logic import WakeUpModelRun, DetectorModelRun, ExpertSystem

with open('data/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
# from gui import MyApp

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

# create buffer both for vad and noise suppression models
in_buffer = np.zeros(BLOCK_LEN).astype('float32')
out_buffer = np.zeros(BLOCK_LEN).astype('float32')


# load DTLN model
dtln_realtime = DTLN(p=1)
# load vad model
vad_model = init_jit_model(VAD_MODEL_PATH)
# load system activation model
run_wake_up = WakeUpModelRun(time_folder='25052022_19-08', best_epoch=14)
# load command detection model
run_detector = DetectorModelRun(time_folder='24052022_20-05', best_epoch=25)
# init expert system
expert_system = ExpertSystem()

audio_is_processed = False
current_part = 0
denoised_speech = []
denoised_audio = np.zeros(BLOCK_LEN_VAD).astype('float32')
system_activated = expert_system.on


def callback(indata, frames, time, status):
    # buffer and states to global
    global dtln_realtime, in_buffer, out_buffer, system_activated
    global denoised_speech, denoised_audio, audio_is_processed, current_part

    if status:
        print(status)

    # DTLN
    in_buffer, out_buffer = dtln_realtime.forward_realtime(indata, in_buffer, out_buffer)
    denoised_audio[:-BLOCK_SHIFT] = denoised_audio[BLOCK_SHIFT:]
    denoised_audio[-BLOCK_SHIFT:] = np.zeros(BLOCK_SHIFT)
    denoised_audio[-BLOCK_SHIFT:] = out_buffer[:BLOCK_SHIFT]

    # VAD
    if current_part == NUM_OF_PARTS-1:
        current_part = -1
        vad_indata_tnsr = torch.FloatTensor(denoised_audio).squeeze()
        if vad_model(vad_indata_tnsr, SAMPLING_RATE).item() > VAD_THRESHOLD:
            audio_is_processed = True
            denoised_speech.extend(denoised_audio)
        elif audio_is_processed == True:
            audio_is_processed = False
            if len(denoised_audio) > 150:
                if not expert_system.on:
                    pred = run_wake_up(denoised_speech)
                    print(pred)
                    run_wake_up.save_file(denoised_speech)
                    if pred > WAKE_UP_THRSH:
                        expert_system.turn_on()
                else:
                    commands = run_detector(denoised_speech)
                    run_detector.save_file(denoised_speech)
                    expert_system(commands[0])
            denoised_speech = []
    current_part+=1


try:
    with sd.InputStream(device=args.input_device,
                   samplerate=SAMPLING_RATE, blocksize=BLOCK_SHIFT,
                   dtype=np.float32, latency=LATENCY,
                   channels=1, callback=callback):

        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()

except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

