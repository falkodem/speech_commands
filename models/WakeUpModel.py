import numpy as np
import torch.nn as nn


class wake_up_model(nn.Module):
    def __init__(self, n_channel=4):
        super(wake_up_model, self).__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(2 * n_channel)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(2 * n_channel, 4 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(4 * n_channel)
        self.act3 = nn.Sigmoid()
        self.pool3 = nn.MaxPool2d((2, 2), 2)
        self.conv4 = nn.Conv2d(4 * n_channel, 8 * n_channel,  kernel_size=(3, 4), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(8 * n_channel)
        self.act4 = nn.Sigmoid()
        self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 8 * n_channel, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        return x