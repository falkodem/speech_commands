from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loaders import SpeechDataset
from torch.utils.data import DataLoader
from utils.utils import *


def create_loaders(path,
                   model_type='wake_up',
                   validation=False,
                   val_size=0.15,
                   test_size=0.15,
                   batch_size=1024,
                   prob=0.5):
    files = sorted(list(Path(path).rglob('*.wav')))
    labels = [path.parent.name for path in files]

    if validation:
        split_size1 = val_size + test_size
        split_size2 = test_size/(val_size+test_size)
    else:
        split_size1 = test_size

    train_files, test_files, train_labels, test_labels = train_test_split(files,
                                                                          labels,
                                                                          test_size=split_size1,
                                                                          stratify=labels,
                                                                          shuffle=True)
    if validation:
        val_files, test_files, val_labels, test_labels = train_test_split(test_files,
                                                                          test_labels,
                                                                          test_size=split_size2,
                                                                          stratify=test_labels,
                                                                          shuffle=True)

    train_dataset = SpeechDataset(train_files, mode='train', prob=prob, model_type=model_type)
    test_dataset = SpeechDataset(test_files, mode='test', prob=prob, model_type=model_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if validation:
        val_dataset = SpeechDataset(val_files, mode='val', prob=prob, model_type=model_type)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None

    return train_loader, val_loader, test_loader


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

    def test_epoch(self, model, epoch, pbar):
        model.eval()
        metric_history_epoch = []
        for X_test, y_test in self.test_loader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            y_pred = model(X_test)

            # add code for multiclass
            print(y_pred.detach().cpu().numpy().astype(int))
            print(y_test.detach().cpu())
            metric_value = self.metric(y_pred.detach().cpu().numpy().astype(int), y_test.detach().cpu())
            # record metric for epoch
            metric_history_epoch.append(metric_value)

            # update progress bar
            pbar.update(self.pbar_update)

        mean_metric_epoch = np.mean(metric_history_epoch)
        self.metric_history.append(mean_metric_epoch)
        print(f"\nTest Epoch: {epoch}\tMetric: {mean_metric_epoch}")


