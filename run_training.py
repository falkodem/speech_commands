import pandas as pd
import torch.nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.autograd import Variable
from utils.utils import *
from train_tools.trainer import Trainer
from data_loaders import LoaderCreator
from models.WakeUpModel import WakeUpModel

MODEL_TYPE = 'wake_up'
log_interval = 40
n_epoch = 50
n_channel = 8
criterion = torch.nn.BCEWithLogitsLoss()
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
                   test_size=0.15,
                   batch_size=256,
                   prob=0.25,
                   from_disc=True)


train_loader, val_loader, test_loader = LC.get_loaders()

# df = {'datka': [], 'class':[]}
#
# for data, label in train_loader:
#     # print(data.shape)
#     df['datka'].append(data[0,0])
#     df['class'].append(label)
#
# import joblib
# joblib.dump(df,'df_ceo_good')


model = WakeUpModel(n_channel=n_channel)
model.to(device)
trainer = Trainer(criterion=criterion,
                  optimizer=optim.Adam(model.parameters(), lr=3.0e-4),
                  train_set=train_loader,
                  test_set=test_loader,
                  metric=roc_auc_score,
                  device=device,
                  model_type=MODEL_TYPE)


# The transform needs to live on the same device as the model and the data.
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        trainer.train_epoch(model, epoch, log_interval, pbar)
        trainer.test_epoch(model, epoch, pbar)
        torch.save(model.state_dict(), SAVE_MODEL_DIR+MODEL_TYPE+'_'+str(epoch)+'.pt')
# print(trainer.metric_history)
results = pd.DataFrame({'loss': trainer.loss_history, 'metric': trainer.metric_history})
results.to_csv(f'./logs/{MODEL_TYPE}_TrainLog.csv')
