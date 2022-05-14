from torchvision import transforms
import torchaudio.backend.soundfile_backend
from scipy.special import softmax
from utils.utils import *
from models.models_init import WakeUpModel, EfficientNet
from utils.preprocessing import MFCC


class WakeUpModelRun:
    def __init__(self, time_folder, best_epoch):
        self.model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
        MODEL_TYPE = 'wake_up'
        self.save_model_dir = SAVE_MODEL_DIR
        self.time_folder = time_folder
        self.best_epoch = best_epoch

        self.model.load_state_dict(
            torch.load(SAVE_MODEL_DIR + MODEL_TYPE + '/' + self.time_folder + '/' + MODEL_TYPE + '_' + 'epoch_' +
                       str(best_epoch) + '.pt'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.extract_feats = torch.nn.Sequential(
            MFCC(fs=SAMPLING_RATE, num_ceps=NUM_CEPS, normalize=True, only_mel=False),
            transforms.Resize((SIZE_Y, SIZE_X))
        )

    def __call__(self, x):
        x = self._preprocess_input(x)
        x = self.extract_feats(x).unsqueeze(0).to(self.device)
        x = self.model(x).detach().cpu().numpy()
        return x

    def _preprocess_input(self, x):
        x = torch.FloatTensor(x)
        return ((x - x.mean()) / x.std()).view(1, 1, -1)

    @staticmethod
    def save_file(x):
        torchaudio.backend.soundfile_backend.save('test.wav', torch.FloatTensor(x).reshape(1, -1),
                                                  SAMPLING_RATE)


class DetectorModelRun:
    def __init__(self, time_folder, best_epoch):
        self.model = EfficientNet()
        MODEL_TYPE = 'detector'
        self.save_model_dir = SAVE_MODEL_DIR
        self.time_folder = time_folder
        self.best_epoch = best_epoch

        self.model.load_state_dict(
            torch.load(SAVE_MODEL_DIR + MODEL_TYPE + '/' + self.time_folder + '/' + MODEL_TYPE + '_' + 'epoch_' +
                       str(best_epoch) + '.pt'))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.extract_feats = torch.nn.Sequential(
            MFCC(fs=SAMPLING_RATE, num_ceps=NUM_CEPS, normalize=True, only_mel=False),
            transforms.Resize((SIZE_Y, SIZE_X))
        )

    def __call__(self, x):
        x = self._preprocess_input(x)
        x = self.extract_feats(x).unsqueeze(0).to(self.device)
        x = self.model(x).detach().cpu().numpy()
        return softmax(x,axis=1)

    def _preprocess_input(self, x):
        x = torch.FloatTensor(x)
        return ((x - x.mean()) / x.std()).view(1, 1, -1)

    @staticmethod
    def save_file(x):
        torchaudio.backend.soundfile_backend.save('test.wav', torch.FloatTensor(x).reshape(1, -1),
                                                  SAMPLING_RATE)
