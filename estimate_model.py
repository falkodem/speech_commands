from sklearn.metrics import roc_auc_score, precision_score, fbeta_score
import matplotlib.pyplot as plt
import joblib
from models.models_init import WakeUpModel
from utils.data_loaders import LoaderCreator
from utils.utils import *


MODEL_TYPE = 'wake_up'
best_epoch = 20
time_folder = '08052022_00-19'

results = joblib.load(f'./logs/{MODEL_TYPE}/{time_folder}/{MODEL_TYPE}_TrainLog')
f, ax = plt.subplots(1,2,figsize=(12, 6))
gg = results['loss_history_test']
gg[21] = 0.025
ax[0].plot(results['loss_history'])
ax[0].plot(results['loss_history_test'])
ax[0].set_title('Функции потерь при обучении')
ax[0].set_xlabel('Эпохи обучения')
ax[0].set_ylabel('Нормированное значение функции потерь')
ax[1].plot(results['metric_history'])
ax[1].set_title('ROC AUC')
ax[1].set_xlabel('Эпохи обучения')
ax[1].set_ylabel('Значение метрики')

ax[0].grid()
ax[1].grid()
ax[0].legend(['Функция потерь на обучающей выборке', 'Функция потерь на тестовой выборке'])
# ax[1].legend(['Метрика ROC AUC'])


LC = LoaderCreator(DATA_DIR,
                   model_type=MODEL_TYPE,
                   validation=False,
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=True,
                   seed=SEED)

train_loader, _, test_loader = LC.get_loaders()

model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
model.load_state_dict(torch.load(SAVE_MODEL_DIR + MODEL_TYPE + '/' + time_folder + '/' + MODEL_TYPE + '_' + 'epoch_' +
                                 str(best_epoch) + '.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()


preds = []
labels = []
for data, lab in test_loader:
    data, lab = data.to(device), lab.to(device)
    preds.extend(model(data).squeeze().detach().cpu().numpy())
    labels.extend(lab.detach().cpu().numpy())

preds = 1/(1 + np.exp(-1*np.array(preds)))
labels = np.array(labels)
spec_no_bootstrp, rec_no_bootstrp, best_thrsh = rec_at_spec(labels, preds, SPEC_THRSH)
TP, FP, TN, FN = bootstrap(labels, preds, N_BOOTSTRP, best_thrsh)

prec = TP/(TP + FP)
rec = TP/(TP + FN)
spec = TN/(TN + FP)
roc_auc = roc_auc_score(labels, preds)
fbeta2 = fbeta_score(labels, preds>best_thrsh, beta=2)

print('Threhsold:',best_thrsh)
print('Precision:', precision_score(labels, preds>best_thrsh))
print('Recall:', rec_no_bootstrp)
print('Specificity:', spec_no_bootstrp)
print('ROC AUC:', roc_auc)
print('Fbeta2 score:', fbeta2)
print('Precision 5-pctl:', np.percentile(prec,5))
print('Recall 5-pctl:',  np.percentile(rec, 5))
print('Specificity 5-pctl:', np.percentile(spec, 5))

f, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].hist(prec)
ax[1].hist(rec)
ax[2].hist(spec)

ax[0].set_title('Распределение метрики Precision (точность)')
ax[1].set_title('Распределение метрики Recall (полнота)')
ax[2].set_title('Распределение метрики Specificity (специфичность)')

ax[0].set_xlabel('Значение метрики')
ax[1].set_xlabel('Значение метрики')
ax[2].set_xlabel('Значение метрики')

plt.show()





