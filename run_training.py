import pandas as pd
import torch.nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from utils.utils import *
from train_tools.trainer import Trainer
from data_loaders import LoaderCreator
from models.WakeUpModel import wake_up_model

MODEL_TYPE = 'wake_up'
log_interval = 20
n_epoch = 50
n_channel = 8
from_disc = True
if from_disc:
    path = DATA_DIR_AUGMENTED
else:
    path = DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LC = LoaderCreator(DATA_DIR,
                   model_type=MODEL_TYPE,
                   validation=False,
                   test_size=0.20,
                   batch_size=1024,
                   prob=0.25,
                   from_disc=True)


train_loader, val_loader, test_loader = LC.get_loaders()

for data, label in test_loader:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(data.shape)
    a = data[label==1]
    # f1, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(a[0,0], annot=False, vmin=0, fmt=".1f")
    # ax.invert_yaxis()
    # plt.show()
    break
# model = wake_up_model(n_channel=n_channel)
# model.to(device)
# trainer = Trainer(criterion=torch.nn.BCEWithLogitsLoss(),
#                   optimizer=optim.Adam(model.parameters(), lr=3.0e-4),
#                   train_set=train_loader,
#                   test_set=test_loader,
#                   metric=roc_auc_score,
#                   device=device,
#                   model_type=MODEL_TYPE)
#
#
# # The transform needs to live on the same device as the model and the data.
# with tqdm(total=n_epoch) as pbar:
#     for epoch in range(1, n_epoch + 1):
#         trainer.train_epoch(model, epoch, log_interval, pbar)
#         trainer.test_epoch(model, epoch, pbar)
#         torch.save(model.state_dict(), SAVE_MODEL_DIR+MODEL_TYPE+'_'+str(epoch)+'.pt')
# print(trainer.metric_history)
# results = pd.DataFrame({'loss': trainer.loss_history, 'metric': trainer.metric_history})
# results.to_csv(f'./logs/{MODEL_TYPE}_TrainLog.csv')
