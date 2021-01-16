import torch
import torch.nn.functional as F
import torch.nn as nn



class BasicChessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU(),
            nn.Conv2d(544, 1024, 3, 2, 1),  # out: 4x4x1024
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, 2, 1), # out: 2x2x2048
            nn.BatchNorm2d(2048),
            nn.ReLU(),

            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 4096, 3, 2, 1), # out: 1x1x4096
            nn.Flatten(),                   # out: 4096
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        return self.layers(x)

class BasicChessCNN2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU(),
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU(),
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU(),
            nn.Conv2d(544, 1024, 3, 2, 1),  # out: 4x4x1024
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 2048, 3, 2, 1), # out: 2x2x2048
            nn.BatchNorm2d(2048),
            nn.ReLU(),

            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            PrintShape(),
            nn.MaxPool2d(2, 2),
            PrintShape(),
            nn.Flatten(),                   # out: 4096
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        return self.layers(x)

class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #print(x.shape)
        return x

class AlphaGoModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(544, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.resblocks = nn.Sequential(*(ResBlock1() for i in range(19)))

        self.head = nn.Sequential(
            nn.Conv2d(256, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 1))

        # conv 2 k 1 s 1
        # bn
        # rlu
        # fc

    def forward(self, x):
        return self.head(self.resblocks(self.first_conv(x)))


class ResBlock1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        first_conv = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv1(first_conv)) + x)



