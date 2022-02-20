import os
import platform
from torchaudio.datasets import SPEECHCOMMANDS
import torch
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__('', download=False)
        # hardcode is needed here cause Windows uses freakin backslashes
        self._path = 'SpeechCommands/speech_commands_v0.02/'
        # #####
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w.replace('\\', '/') not in excludes]

# creating subsets
train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

####################################################
labels = sorted(list(set(tqdm(datapoint[2] for datapoint in test_set))))
print(len(labels))


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

print('dataloader')
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
waveform_last, *_ = train_set[-1]

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

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
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = M5(n_input=waveform_last.shape[0], n_output=len(labels))
model.to(device)
print(model)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

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
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()
#################################################################################################################
# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");

# # print(len(train_set))
# # labels = list(datapoint[2] for datapoint in train_set)
# # print(labels.count('duza'))
#
# labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
#
# def label_to_index(word):
#   if word == 'stop':
#     return 1
#   else:
#     return 0
#
#
# def index_to_label(index):
#     # Return the word corresponding to the index in labels
#     # This is the inverse of label_to_index
#     return labels[index]
#
# def pad_sequence(batch):
#     # Make all tensor in a batch the same length by padding with zeros
#     batch = [item.t() for item in batch]
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
#     return batch.permute(0, 2, 1)
#
#
# def collate_fn(batch):
#
#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number
#
#     tensors, targets = [], []
#
#     # Gather in lists, and encode labels as indices
#     for waveform, _, label, *_ in batch:
#         tensors += [waveform]
#         targets += [label_to_index(label)]
#
#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence(tensors)
#     targets = torch.stack(targets)
#
#     return tensors, targets
#
#
# batch_size = 256
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device == "cuda":
#     num_workers = 1
#     pin_memory = True
# else:
#     num_workers = 0
#     pin_memory = False
#
# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )
#
# val_loader = torch.utils.data.DataLoader(
#     val_set,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )
#
# test_loader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )