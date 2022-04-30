from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, fbeta_score, \
    precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models.WakeUpModel import WakeUpModel
from data_loaders import LoaderCreator
from utils.utils import *


MODEL_TYPE = 'wake_up'
best_epoch = 5

results = pd.read_csv(f'./logs/{MODEL_TYPE}_TrainLog.csv')
f, ax = plt.subplots(1,2,figsize=(12, 6))
ax[0].plot(results['loss'])
ax[0].plot(results['loss_test'])
ax[1].plot(results['metric'])

ax[0].grid()
ax[1].grid()
ax[0].legend(['Функция потерь на обучающей выборке', 'Функция потерь на тестовой выборке'])
ax[1].legend(['Метрика'])

# plt.show()

LC = LoaderCreator(DATA_DIR,
                   model_type=MODEL_TYPE,
                   validation=False,
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=True,
                   seed=SEED)

_, _, test_loader = LC.get_loaders()

model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
model.load_state_dict(torch.load(SAVE_MODEL_DIR + MODEL_TYPE + '_' + str(best_epoch) + '.pt'))
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

print(roc_auc_score(labels, preds))
prec, rec, thrsh = precision_recall_curve(labels, preds)
f = plt.figure(figsize=(12, 6))
plt.plot(prec)
plt.plot(rec)
plt.plot(thrsh)
plt.grid()
print(precision_score(labels, preds>0.55))
print(recall_score(labels, preds>0.55))
print(fbeta_score(labels, preds>0.55, beta = 2))
plt.show()





