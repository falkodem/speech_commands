import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, fbeta_score, accuracy_score, confusion_matrix,\
    matthews_corrcoef
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from scipy.special import softmax
from models.models_init import WakeUpModel, EfficientNet
from utils.data_loaders import LoaderCreator
from utils.utils import *

# wake_up or detector
MODEL_TYPE = 'wake_up'

if MODEL_TYPE == 'wake_up':
    # time_folder = '08052022_00-19'
    # новая
    # time_folder = '23052022_23-37'
    time_folder = '25052022_19-08'
    best_epoch = 14
else:
    # time_folder = '10052022_02-05'
    # новая
    time_folder = '24052022_20-05'
    best_epoch = 25


results = joblib.load(f'./logs/{MODEL_TYPE}/{time_folder}/{MODEL_TYPE}_TrainLog')
f, ax = plt.subplots(1,2,figsize=(12, 6))
gg = results['loss_history_test']
ax[0].plot(results['loss_history'])
ax[0].plot(results['loss_history_test'])
ax[0].set_title('Функции потерь при обучении')
ax[0].set_xlabel('Эпохи обучения')
ax[0].set_ylabel('Нормированное значение функции потерь')
ax[1].plot(results['metric_history'])
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Эпохи обучения')
ax[1].set_ylabel('Значение метрики')

ax[0].grid()
ax[1].grid()
ax[0].legend(['Функция потерь на обучающей выборке', 'Функция потерь на валидационной выборке'])
# ax[1].legend(['Метрика ROC AUC'])


# LC = LoaderCreator(DATA_DIR,
#                    model_type=MODEL_TYPE,
#                    validation=False,
#                    test_size=TEST_SIZE,
#                    batch_size=BATCH_SIZE,
#                    prob=PROB,
#                    from_disc=True,
#                    seed=SEED)
LC = LoaderCreator(DATA_DIR,
                   model_type=MODEL_TYPE,
                   validation=True,
                   val_size=VAL_SIZE,
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=False,
                   seed=SEED)

train_loader, _, test_loader = LC.get_loaders()

if MODEL_TYPE == 'wake_up':
    model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
else:
    model = EfficientNet()
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

labels = np.array(labels)

if MODEL_TYPE == 'wake_up':
    spec_no_bootstrp, rec_no_bootstrp, best_thrsh = rec_at_spec(labels, preds, SPEC_THRSH)
    TP, FP, TN, FN = bootstrap(labels, preds, N_BOOTSTRP, best_thrsh)
    prec = TP/(TP + FP)
    rec = TP/(TP + FN)
    spec = TN/(TN + FP)
    roc_auc = roc_auc_score(labels, preds)
    fbeta2 = fbeta_score(labels, preds>best_thrsh, beta=0.5)

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
    ax[0].hist(prec, bins=20)
    ax[1].hist(rec, bins=20)
    ax[2].hist(spec, bins=20)

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].set_title('Распределение метрики Precision (точность)')
    ax[1].set_title('Распределение метрики Recall (полнота)')
    ax[2].set_title('Распределение метрики Specificity (специфичность)')

    ax[0].set_xlabel('Значение метрики')
    ax[1].set_xlabel('Значение метрики')
    ax[2].set_xlabel('Значение метрики')

    conf_matrix = confusion_matrix(labels, preds > best_thrsh)
    g = plt.figure(figsize=(16,8))
    sns.heatmap(conf_matrix,annot=True)
    plt.show()
    print(labels.shape)
else:
    preds1 = np.argmax(preds, axis=1)
    preds2 = softmax(preds,axis=1)
    print('Accuracy', accuracy_score(labels, preds1))
    conf_matrix = confusion_matrix(labels, preds1)
    print(conf_matrix.shape)
    g = plt.figure(figsize=(16,8))
    sns.heatmap(conf_matrix, annot=True, cmap='gray_r')
    print("Precision:", precision_score(labels, preds1, average='macro'))
    print("Recall:", precision_score(labels, preds1, average='macro'))
    print('Fbeta2 score:', fbeta_score(labels, preds1, beta=0.5, average='macro'))
    print('Matthew corrcoef:', matthews_corrcoef(labels, preds1))
    print('ROC AUC:', roc_auc_score(labels, preds2, average='macro', multi_class='ovr'))

    plt.show()





