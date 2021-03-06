import numpy as np
import torch.nn as nn
import torch


class WakeUpModel(nn.Module):
    def __init__(self, n_channel=4):
        super(WakeUpModel, self).__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.act1 = nn.ReLU()
        self.conv1b = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1b = nn .BatchNorm2d(2 * n_channel)
        self.act1b = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2), 2)

        self.conv2 = nn.Conv2d(2 * n_channel, 4 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(4 * n_channel)
        self.act2 = nn.ReLU()
        self.conv2b = nn.Conv2d(4 * n_channel, 8 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2b = nn.BatchNorm2d(8 * n_channel)
        self.act2b = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2), 2)

        self.conv3 = nn.Conv2d(8 * n_channel, 16 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16 * n_channel)
        self.act3 = nn.ReLU()
        self.conv3b = nn.Conv2d(16 * n_channel, 32 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3b = nn.BatchNorm2d(32 * n_channel)
        self.act3b = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2, 2), 2)

        self.conv4 = nn.Conv2d(32 * n_channel, 64 * n_channel,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64 * n_channel)
        self.act4 = nn.ReLU()
        self.conv4b = nn.Conv2d(64 * n_channel, 96 * n_channel,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4b = nn.BatchNorm2d(96 * n_channel)
        self.act4b = nn.ReLU()
        self.pool4 = nn.MaxPool2d((2, 2), 2)

        self.conv5 = nn.Conv2d(96 * n_channel, 128 * n_channel,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(128 * n_channel)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((2, 2), 2)

        self.fc1 = nn.Linear(1 * 4 * 128 * n_channel, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.act1b(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.act2b(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.act3b(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.act4b(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.sig(x)
        return x

class WakeUpModel2(nn.Module):
    def __init__(self, n_channel=4):
        super(WakeUpModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(2 * n_channel)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(2 * n_channel, 4 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(4 * n_channel)
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), 2)
        self.conv4 = nn.Conv2d(4 * n_channel, 8 * n_channel,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(8 * n_channel)
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2 * 8 * 8 * n_channel, 1)
        self.sig = nn.Sigmoid()

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
        x = self.sig(x)
        return x


class DetectorModel(nn.Module):
    def __init__(self, n_channel=4):
        super(DetectorModel, self).__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(2 * n_channel)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(2 * n_channel, 4 * n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(4 * n_channel)
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), 2)
        self.conv4 = nn.Conv2d(4 * n_channel, 8 * n_channel,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(8 * n_channel)
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2 * 8 * 8 * n_channel, 28)

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


def EfficientNet():
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

    efficientnet.stem = torch.nn.Sequential(
                                      torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                      torch.nn.BatchNorm2d(32),
                                      torch.nn.SiLU(inplace=True)
    )
    efficientnet.classifier = torch.nn.Sequential(
                                      torch.nn.AdaptiveAvgPool2d(output_size=1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(in_features=1280, out_features=28, bias=True)
    )
    return efficientnet
