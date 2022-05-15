import pickle
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
        torchaudio.backend.soundfile_backend.save('../test_wakeup.wav', torch.FloatTensor(x).reshape(1, -1),
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
        torchaudio.backend.soundfile_backend.save('../test_detector.wav', torch.FloatTensor(x).reshape(1, -1),
                                                  SAMPLING_RATE)


class ExpertSystem:

    def __init__(self):
        self.on = False

        with open('data/label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

        self.request_lvl = 1

        self.lvl_1_cmd_pool = ['mode', 'information', 'input', 'notebook', 'chassis_on', 'chassis_off', 'cancel']
        self.lvl_1_cmd_pool_idx = self.le.transform(self.lvl_1_cmd_pool)
        self.lvl_1_cmd = None

        self.chassis_cmd_pool = ['yes', 'no', 'cancel', 'correct']
        self.chassis_cmd_pool_idx = self.le.transform(self.chassis_cmd_pool)
        self.mode_cmd_pool = ['autopilot', 'manual', 'cancel']
        self.mode_cmd_pool_idx = self.le.transform(self.mode_cmd_pool)
        self.information_cmd_pool = ['coordinates', 'fuel', 'height', 'speed', 'temperature', 'cancel']
        self.information_cmd_pool_idx = self.le.transform(self.information_cmd_pool)
        self.input_cmd_pool = ['coordinates', 'height', 'speed', 'temperature', 'cancel']
        self.input_cmd_pool_idx = self.le.transform(self.input_cmd_pool)
        self.notebook_cmd_pool = ['yes', 'no', 'correct', 'cancel']
        self.notebook_cmd_pool_idx = self.le.transform(self.notebook_cmd_pool)
        self.lvl_2_cmd = None

        self.get_input_cmd_pool = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'cancel', 'correct', 'yes', 'no']
        self.get_input_cmd_pool_idx = self.le.transform(self.get_input_cmd_pool)

        self.system_states = {'coordinates': 100,
                              'fuel': 100,
                              'height': 6000,
                              'speed': 300,
                              'temperature': 20}
        self.system_devices = {'chassis': False}

        self.input_states = []

    def __call__(self, indata):
        if self.request_lvl == 1:
            self._lvl_1_proc(indata)
        elif self.request_lvl == 2:
            self._lvl_2_proc(indata)
        elif self.request_lvl == 3:
            self._lvl_3_proc(indata)

    def _lvl_1_proc(self, indata):
        # getting probs of lvl 1 command
        indata_lvl_1 = [indata[i] for i in self.lvl_1_cmd_pool_idx]
        if np.max(indata_lvl_1) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Команда не распознана.')
        else:
            chosen_cmd_idx = self.lvl_1_cmd_pool_idx[np.argmax(indata_lvl_1)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])[0]
            if chosen_cmd == 'mode':
                self.request_lvl = 2
                self.lvl_2_func = self._mode
            # в файл вывода аппендится "режим" - kivy выводит "выберете режим"
                print('Выбор режима:')
            elif chosen_cmd == 'information':
                self.request_lvl = 2
                self.lvl_2_func = self._information
                # в файл вывода аппендится "информация" - kivy выводит "какую информацию вывести"
                print('Какую информацию вывести?')
            elif chosen_cmd == 'input':
                self.request_lvl = 2
                self.lvl_2_func = self._input
                print('Какой параметр изменить?')
                # в файл вывода аппендится "ввод" - kivy выводит "какую информацию ввести"
            elif chosen_cmd == 'notebook':
                self.request_lvl = 2
                self.lvl_2_func = self._notebook
                # в файл вывода аппендится "блокнот" - kivy выводит "Начать запись?" хотя ????
            elif chosen_cmd == 'chassis_on':
                if self.system_devices['chassis']:
                    self.request_lvl = 1
                    print('Шасси уже выпущены')
                else:
                    print('Вы уверены, что хотите выпустить шасси?')
                    self.request_lvl = 2
                    self.lvl_2_func = self._chassis
            elif chosen_cmd == 'chassis_off':
                # в файл вывода аппендится "шасси убраны" - kivy переводит индикатор шасси в OFF
                if not self.system_devices['chassis']:
                    self.request_lvl = 1
                    print('Шасси уже убраны')
                else:
                    self.request_lvl = 2
                    print('Вы уверены, что хотите убрать шасси?')
                    self.lvl_2_func = self._chassis
            else:
                self.turn_off()

            self.lvl_1_cmd = chosen_cmd

    def _lvl_2_proc(self, indata):
        # getting probs of lvl 2 command
        self.lvl_2_func(indata)

    def _lvl_3_proc(self, indata):
        self.lvl_3_func(indata)

    def _chassis(self, indata):
        pool_idx = self.chassis_cmd_pool_idx
        indata_lvl_2 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_2) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Команда режима не распознанана. Повторите ввод')
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_2)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])
            if self.system_devices['chassis']:
                if (chosen_cmd == 'yes') | (chosen_cmd == 'correct'):
                    self.request_lvl = 1
                    print('Шасси убраны')
                    self.system_devices['chassis'] = False
                    self.turn_off()
                    # в файл вывода аппендится "шасси убраны" - kivy переводит индикатор шасси в OFF
                elif (chosen_cmd == 'no') | (chosen_cmd == 'cancel'):
                    self.request_lvl = 1
                    print('Отмена операции. Ввод первой команды')
            else:
                if (chosen_cmd == 'yes') | (chosen_cmd == 'correct'):
                    self.request_lvl = 1
                    print('Шасси выпущены')
                    self.system_devices['chassis'] = True
                    self.turn_off()
                    # в файл вывода аппендится "шасси выпущены" - kivy переводит индикатор шасси в ON
                elif (chosen_cmd == 'no') | (chosen_cmd == 'cancel'):
                    self.request_lvl = 1
                    print('Отмена операции. Ввод первой команды')

    def _mode(self,indata):
        pool_idx = self.mode_cmd_pool_idx
        indata_lvl_2 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_2) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Команда режима не распознанана. Повторите ввод')
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_2)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])
            if chosen_cmd == 'autopilot':
                self.request_lvl = 1
                print('Автопилот')
                self.turn_off()
                # в файл вывода аппендится "автопилот" - kivy переводит индикатор автопилота в ON
            elif chosen_cmd == 'manual':
                self.request_lvl = 1
                print('Ручное управление')
                self.turn_off()
                # в файл вывода аппендится "информация" - kivy переводит индикатор автопилота в OFF
            else:
                # возвращение на предыдущий шаг
                # в файл вывода аппендится "Ввод первой команды" - kivy подает сигнал отмена и выводит "Ввод первой команды"
                self.request_lvl = 1
                print('Отмена операции. Ввод первой команды')
                # ???

    def _information(self,indata):
        pool_idx = self.information_cmd_pool_idx
        indata_lvl_2 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_2) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Команда запроса информации не распознанана. Повторите ввод')
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_2)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])[0]
            if chosen_cmd == 'coordinates':
                self.request_lvl = 1
                print('Координаты:', self.system_states['coordinates'])
                self.turn_off()
                # в файл вывода аппендится "координаты" - kivy выводит "координаты: {value}
            elif chosen_cmd == 'fuel':
                self.request_lvl = 1
                print('Топливо:', self.system_states['fuel'])
                self.turn_off()
                # в файл вывода аппендится "топливо" - kivy выводит "топливо: {value}
            elif chosen_cmd == 'height':
                self.request_lvl = 1
                print('Высота:', self.system_states['height'])
                self.turn_off()
                # в файл вывода аппендится "высота" - kivy выводит "высота: {value}
            elif chosen_cmd == 'speed':
                self.request_lvl = 1
                print('Скорость:', self.system_states['speed'])
                self.turn_off()
                # в файл вывода аппендится "скорость" - kivy выводит "скорость: {value}
            elif chosen_cmd == 'temperature':
                self.request_lvl = 1
                print('Температура:', self.system_states['temperature'])
                self.turn_off()
                # в файл вывода аппендится "температура" - kivy выводит "температура: {value}
            else:
                # возвращение на предыдущий шаг
                # в файл вывода аппендится "Ввод первой команды" - kivy подает сигнал отмена и выводит "Ввод первой команды"
                self.request_lvl = 1
                print('Отмена операции. Ввод первой команды')
                # ???

    def _input(self, indata):
        pool_idx = self.input_cmd_pool_idx
        indata_lvl_2 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_2) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Команда ввода информации не распознанана. Повторите ввод')
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_2)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])[0]
            if chosen_cmd == 'coordinates':
                self.request_lvl = 3
                print('Задайте координаты ...')
                self.lvl_3_func = self._get_input
                # в файл вывода аппендится "координаты" - kivy выводит "координаты: {value}
            elif chosen_cmd == 'height':
                self.request_lvl = 3
                print('Задайте высоту ...')
                self.lvl_3_func = self._get_input
                # в файл вывода аппендится "высота" - kivy выводит "высота: {value}
            elif chosen_cmd == 'speed':
                self.request_lvl = 3
                print('Задайте скорость ...')
                self.lvl_3_func = self._get_input
                # в файл вывода аппендится "скорость" - kivy выводит "скорость: {value}
            elif chosen_cmd == 'temperature':
                self.request_lvl = 3
                print('Задайте температуру ...')
                self.lvl_3_func = self._get_input
                # в файл вывода аппендится "температура" - kivy выводит "температура: {value}
            else:
                # возвращение на предыдущий шаг
                # в файл вывода аппендится "Ввод первой команды" - kivy подает сигнал отмена и выводит "Ввод первой команды"
                self.request_lvl = 1
                print('Отмена операции. Ввод первой команды')
                # ???
            self.lvl_2_cmd = chosen_cmd

    def _notebook(self, indata):
        pool_idx = self.notebook_cmd_pool_idx
        indata_lvl_2 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_2) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            pass
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_2)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])
            if (chosen_cmd == 'yes') | (chosen_cmd == 'correct'):
                self.request_lvl = 3
                # в файл вывода аппендится "запись" - kivy переводит индикатор записи в ON
            elif (chosen_cmd == 'no') | (chosen_cmd == 'cancel'):
                # возвращение на предыдущий шаг
                # в файл вывода аппендится "Ввод первой команды" - kivy подает сигнал отмена и выводит "Ввод первой команды"
                self.request_lvl = 1
                # ???

    def _get_input(self, indata):
        pool_idx = self.get_input_cmd_pool_idx
        indata_lvl_3 = [indata[i] for i in pool_idx]
        if np.max(indata_lvl_3) < CMD_ACCEPT_THRSH:
            # в файл вывода аппендится "команда не распознана" - kivy подает сигнал и пишет "команда не распознана"
            print('Цифра не распознана. Повторите ввод')
        else:
            chosen_cmd_idx = pool_idx[np.argmax(indata_lvl_3)]
            chosen_cmd = self.le.inverse_transform([chosen_cmd_idx])[0]
            if chosen_cmd == 'cancel':
                # возвращение на предыдущий шаг
                # в файл вывода аппендится "Ввод первой команды" - kivy подает сигнал отмена и выводит "Ввод первой команды"
                self.request_lvl = 2
                self.input_states = []
                print('Отмена операции. Выберите параметр, который должен быть изменен')
                # ???
            elif chosen_cmd == 'correct':
                self.system_states[self.lvl_2_cmd] = int(''.join(self.input_states))
                print(f'Ввод параметра "{self.lvl_2_cmd}" завершен. Значение: {self.system_states[self.lvl_2_cmd]}')
                self.turn_off()
            else:
                self.input_states.append(chosen_cmd)
                print(f'Введено: {self.input_states}')


    def turn_on(self):
        print('------------------------')
        print('| Система активирована |')
        print('------------------------')
        self.on = True

    def turn_off(self):
        print('-------------------------------------')
        print('| Система переведена в спящий режим |')
        print('-------------------------------------')
        self.request_lvl = 1
        self.on = False