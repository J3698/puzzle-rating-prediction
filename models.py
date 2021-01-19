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
        return F.relu(self.bn2(self.conv2(first_conv)) + x)


class AlphaGoModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU()
        )

        self.resblocks1 = nn.Sequential(*(ResBlock2(544) for i in range(8)))
        self.downsample = ResBlock2(544, half = True)
        self.resblocks2 = nn.Sequential(*(ResBlock2(1088) for i in range(3)))

        self.head = nn.Sequential(
            PrintShape(),
            nn.Conv2d(1088, 2, 1, 1),
            PrintShape(),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 1))

        # conv 2 k 1 s 1
        # bn
        # rlu
        # fc

    def forward(self, x):
        out = self.resblocks1(self.first_conv(x))
        return self.head(self.resblocks2(self.downsample(out)))


class ResBlock2(nn.Module):
    def __init__(self, in_c, half = False):
        super().__init__()

        out_c = 2 * in_c if half else in_c
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2 if half else 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        if half:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, 2, 0),
                    nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        first_conv = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(first_conv)) + self.shortcut(x))


class AlphaGoModelG(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.GELU()
        )

        self.resblocks1 = nn.Sequential(*(ResBlockG(544) for i in range(8)))
        self.downsample = ResBlockG(544, half = True)
        self.resblocks2 = nn.Sequential(*(ResBlockG(1088) for i in range(3)))

        self.head = nn.Sequential(
            PrintShape(),
            nn.Conv2d(1088, 2, 1, 1),
            PrintShape(),
            nn.BatchNorm2d(2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32, 1))

        # conv 2 k 1 s 1
        # bn
        # rlu
        # fc

    def forward(self, x):
        out = self.resblocks1(self.first_conv(x))
        return self.head(self.resblocks2(self.downsample(out)))


class ResBlockG(nn.Module):
    def __init__(self, in_c, half = False):
        super().__init__()

        out_c = 2 * in_c if half else in_c
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2 if half else 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        if half:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, 2, 0),
                    nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        first_conv = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(self.bn2(self.conv2(first_conv)) + self.shortcut(x))
