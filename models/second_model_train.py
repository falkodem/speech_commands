import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.utils import *
from train_tools.trainer import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)


class WakeupModel(nn.Module):
    def __init__(self, n_input=(SIZE_Y, SIZE_X), n_output=2, n_channel=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_channel, (3, 3), (1, 1), 1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(n_channel, 2 * n_channel, (3, 3), (1, 1), 1)
        self.bn2 = nn.BatchNorm2d(2 * n_channel)
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(2 * n_channel, 4 * n_channel, (3, 3), (1, 1), 1)
        self.bn3 = nn.BatchNorm2d(4 * n_channel)
        self.pool3 = nn.MaxPool2d((2, 2), 2)
        self.conv4 = nn.Conv2d(4 * n_channel, 8 * n_channel,  (3, 4), (1, 1), 1)
        self.bn4 = nn.BatchNorm2d(8 * n_channel)
        self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 8 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        return x


wake_up = WakeupModel()
wake_up.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(wake_up.parameters(), lr=0.01, weight_decay=0.0001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10


def train_epoch(model, epoch, log_interval):
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        # apply transform and model on whole batch directly on device
        y_pred = model(X_batch)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        metric = roc_auc_score(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(X_batch)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        loss_epoch += loss.item()
        # record metric
        metric_history_epoch.append(metric)

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
loss_history = []

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device

        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")



# The transform needs to live on the same device as the model and the data.
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval )
        # test(model, epoch)
        scheduler.step()
