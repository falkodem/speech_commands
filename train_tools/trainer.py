import os
from datetime import datetime

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib

from utils.utils import *


def train_loop(trainer, model, model_type, n_epoch=20, log_interval=20, early_stop=5):
    now = datetime.now().strftime('%d%m%Y_%H-%M')
    os.mkdir(f'{SAVE_MODEL_DIR}{model_type}/{now}')

    overfit_epochs_cnt = 0
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(0, n_epoch):
            trainer.train_epoch(model, epoch, log_interval, pbar)
            trainer.test_epoch(model, epoch, pbar)

            if epoch == 0:
                best_loss = trainer.loss_history_test[-1]
                best_metric = trainer.metric_history[-1]
                best_epoch = epoch
                best_states = model.state_dict()
            else:
                if trainer.loss_history_test[-1] > best_loss:
                    overfit_epochs_cnt += 1
                    if overfit_epochs_cnt >= early_stop:
                        print(f'\nBest epoch: {best_epoch}. Training stopped.')
                        break
                else:
                    best_loss = trainer.loss_history_test[-1]
                    best_metric = trainer.metric_history[-1]
                    best_epoch = epoch
                    best_states = model.state_dict()
                    overfit_epochs_cnt = 0
                    torch.save(model.state_dict(),
                               f'{SAVE_MODEL_DIR}{model_type}/{now}/{model_type}_epoch_{str(epoch)}.pt')

            print('Epoch: {:1}\tTrain_loss {:7}\tTest_loss: {:7}\tMetric: {:3}\tBest_epoch: {:1}'
                  .format(epoch, trainer.loss_history[-1], trainer.loss_history_test[-1], trainer.metric_history[-1],
                          best_epoch))

    results = {'loss_history': trainer.loss_history,
               'loss_history_test': trainer.loss_history_test,
               'metric_history': trainer.metric_history,
               'best_loss': best_loss,
               'best_metric': best_metric,
               'best_epoch': best_epoch,
               'best_states': best_states}

    os.mkdir(f'./logs/{model_type}/{now}')
    joblib.dump(results, f'./logs/{model_type}/{now}/{model_type}_TrainLog')

    return results


class Trainer:
    def __init__(self,
                 criterion,
                 optimizer,
                 train_set,
                 test_set,
                 metric,
                 device,
                 model_type):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_set
        self.test_loader = test_set
        self.metric = metric
        self.model_type = model_type
        self.loss_history = []
        self.loss_history_test = []
        self.metric_history = []
        self.device = device
        self.pbar_update = 1 / (len(self.train_loader) + len(self.test_loader))

        if (self.model_type != 'wake_up') & (self.metric != accuracy_score):
            raise ValueError("Only accuracy-metric is supported for multiclass classification (second model)")

    def train_epoch(self, model, epoch, log_interval, pbar):
        loss_epoch = 0
        model.train()
        df_len = self.train_loader.__len__()
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = model(X_batch)

            if self.model_type == 'wake_up':
                loss = self.criterion(y_pred, y_batch.unsqueeze(1).float())
            else:
                loss = self.criterion(y_pred, y_batch)
            loss.backward()

            self.optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(f"\nTrain Epoch: {epoch} [{batch_idx * len(X_batch)}/{len(self.train_loader.dataset)}"
                      f" ({100. * batch_idx / len(self.train_loader):.0f}%)]")

            # update progress bar
            pbar.update(self.pbar_update)
            # accumulate loss for epoch
            loss_epoch += loss.item() / df_len
        self.loss_history.append(loss_epoch)

    def test_epoch(self, model, epoch, pbar):
        model.eval()
        metric_history_epoch = []
        loss_epoch_test = 0
        df_len = self.test_loader.__len__()
        for X_test, y_test in self.test_loader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            y_pred = model(X_test)
            # add code for multiclass
            # print(y_pred[y_test == 1][:7].flatten())
            # print(y_pred[y_test == 0][:7].flatten())

            if self.model_type == 'wake_up':
                metric_value = self.metric(y_test.detach().cpu(), y_pred.detach().cpu().numpy())
                loss_test = self.criterion(y_pred, y_test.unsqueeze(1).float())
            else:
                metric_value = self.metric(y_test.detach().cpu(), np.argmax(y_pred.detach().cpu().numpy(), axis=1))
                loss_test = self.criterion(y_pred, y_test)

            # record metric for epoch
            metric_history_epoch.append(metric_value)
            # record loss test for epoch
            loss_epoch_test += loss_test.item() / df_len
            # update progress bar
            pbar.update(self.pbar_update)

        self.loss_history_test.append(loss_epoch_test)
        mean_metric_epoch = np.mean(metric_history_epoch)
        self.metric_history.append(mean_metric_epoch)

