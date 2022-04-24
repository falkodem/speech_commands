import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.utils import *
from train_tools.trainer import create_loaders, Trainer
from models.WakeUpModel import wake_up_model

MODEL_TYPE = 'wake_up'
log_interval = 20
n_epoch = 2
n_channel = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_loader, val_loader, test_loader = create_loaders(DATA_DIR,
                                                       model_type=MODEL_TYPE,
                                                       validation=False,
                                                       test_size=0.3,
                                                       batch_size=256,
                                                       prob=0.5)


model = wake_up_model(n_channel=n_channel)
model.to(device)
trainer = Trainer(criterion=torch.nn.BCEWithLogitsLoss(),
                  optimizer=optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001),
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

results = pd.DataFrame({'loss': trainer.loss_history, 'matric': trainer.metric_history})
results.to_csv(f'./logs/{MODEL_TYPE}_TrainLog.csv')
