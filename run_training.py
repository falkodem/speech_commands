import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils.utils import *
from train_tools.trainer import Trainer, train_loop
from data_loaders import LoaderCreator
from models.WakeUpModel import WakeUpModel

MODEL_TYPE = 'wake_up'

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
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=from_disc,
                   seed=SEED)

train_loader, val_loader, test_loader = LC.get_loaders()

model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
model.to(device)
trainer = Trainer(criterion=criterion,
                  optimizer=optim.Adam(model.parameters(), lr=3.0e-4),
                  train_set=train_loader,
                  test_set=test_loader,
                  metric=roc_auc_score,
                  device=device,
                  model_type=MODEL_TYPE)

results = train_loop(trainer=trainer,
                     model=model,
                     model_type=MODEL_TYPE,
                     n_epoch=N_EPOCH,
                     log_interval=LOG_INTERVAL)

