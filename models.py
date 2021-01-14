import torch
import torch.nn as nn



class BasicChessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(544, 544, 3, 1, 1),
            nn.BatchNorm2d(544),
            nn.ReLU(),
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

