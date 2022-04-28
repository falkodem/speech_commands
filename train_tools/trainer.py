from tqdm import tqdm

from data_loaders import SpeechDataset
from torch.utils.data import DataLoader
from utils.utils import *

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
        self.metric_history = []
        self.device = device
        self.pbar_update = 1 / (len(self.train_loader) + len(self.test_loader))


        if (self.model_type != 'wake_up') & (self.metric != accuracy):
            raise ValueError("Only accuracy-metric is supported for multiclass classification (second model)")

    def train_epoch(self, model, epoch, log_interval, pbar):
        loss_epoch = 0
        model.train()
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = self.criterion(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()

            self.optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(f"\nTrain Epoch: {epoch} [{batch_idx * len(X_batch)}/{len(self.train_loader.dataset)}"
                      f" ({100. * batch_idx / len(self.train_loader):.0f}%)]")

            # update progress bar
            pbar.update(self.pbar_update)
            # accumulate loss for epoch
            loss_epoch += loss.item()
        self.loss_history.append(loss_epoch)
        print(f"\nEpoch: {epoch}\t\tLoss: {loss_epoch}")

    def test_epoch(self, model, epoch, pbar):
        model.eval()
        metric_history_epoch = []
        for X_test, y_test in self.test_loader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            y_pred = model(X_test)

            # add code for multiclass
            metric_value = self.metric(y_test.detach().cpu(), y_pred.detach().cpu().numpy())
            # record metric for epoch
            metric_history_epoch.append(metric_value)

            # update progress bar
            pbar.update(self.pbar_update)

        mean_metric_epoch = np.mean(metric_history_epoch)
        self.metric_history.append(mean_metric_epoch)
        print(f"\nEpoch: {epoch}\t\tMetric: {mean_metric_epoch}")


