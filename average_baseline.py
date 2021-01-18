#!/usr/bin/env python3

import argparse


from math import inf
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import *
from dataset import get_dataloaders
from train_predictor import scaled_l2_loss




BATCH_SIZE = 2048
TRUNCATION = inf#10000


def main():
    train_dataloader, val_dataloader, _ = get_dataloaders(BATCH_SIZE, truncation = TRUNCATION)

    with torch.no_grad():
        ys = []
        for _, y in tqdm.tqdm(train_dataloader):
            ys.append(y[:, 0].mean().item())
        mean = np.mean(ys)
        print(mean)

        losses_temp = 0
        losses = []
        correct = 0
        total = 0
        for i, (_, y) in enumerate(tqdm.tqdm(train_dataloader)):
            loss = scaled_l2_loss(torch.ones((y.shape[0], 1))*mean, y, reduction = 'none')
            loss_reduced = loss.mean()
            correct += (loss < 1).sum()
            total += len(y)
            losses.append(loss_reduced.item())
            losses_temp += loss_reduced.item()


            if i % 60 == 0:
                if i != 0:
                    print(f"{losses_temp / 60:.3f}")
                losses_temp = 0
        print(np.mean(losses))

        print(correct/total)




if __name__ == "__main__":
    main()
