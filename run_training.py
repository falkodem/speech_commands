import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.utils import *
from train_tools.trainer import Trainer, train_loop
from utils.data_loaders import LoaderCreator
from models.models_init import EfficientNet, WakeUpModel

MODEL_TYPE = 'wake_up'
from_disc = False

if from_disc:
    path = DATA_DIR_AUGMENTED
else:
    path = DATA_DIR

if MODEL_TYPE == 'wake_up':
    criterion = torch.nn.BCELoss()
    metric = roc_auc_score
    model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
else:
    criterion = torch.nn.CrossEntropyLoss()
    metric = accuracy_score
    model = EfficientNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

LC = LoaderCreator(path,
                   model_type=MODEL_TYPE,
                   validation=True,
                   val_size=VAL_SIZE,
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=from_disc,
                   seed=SEED)

train_loader, val_loader, test_loader = LC.get_loaders()

trainer = Trainer(criterion=criterion,
                  optimizer=optim.Adam(model.parameters(), lr=1.0e-4),
                  train_set=train_loader,
                  test_set=val_loader,
                  metric=metric,
                  device=device,
                  model_type=MODEL_TYPE)

results = train_loop(trainer=trainer,
                     model=model,
                     model_type=MODEL_TYPE,
                     n_epoch=N_EPOCH,
                     log_interval=LOG_INTERVAL)

