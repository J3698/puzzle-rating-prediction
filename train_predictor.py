#!/usr/bin/env python3

# imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import *

TRUNCATION = 1

class PuzzleDataset(Dataset):
    def __init__(self, puzzles_file, ratings_file, shapes_file):
        """
        reader = csv.reader(open('filename.csv', 'r'))
        shapes = {k : eval(v) for k, v in reader}
        """

        self.puzzles = np.memmap(puzzles_file, mode="r", dtype=np.int8, shape=(797589, 544, 8, 8))
        self.ratings = np.memmap(ratings_file, mode="r", dtype=np.float32, shape=(797589, 2))


    def __len__(self):
        return min(len(self.puzzles), TRUNCATION)


    def __getitem__(self, idx):
        return torch.tensor(self.puzzles[idx].copy(), dtype=torch.float32),\
               torch.tensor(self.ratings[idx].copy(), dtype=torch.float32)


def main():
    dataset = PuzzleDataset("./data/puzzles.dat", "./data/ratings.dat", "./data/shapes.csv")
    train_dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)
    model = BasicChessCNN()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.3)

    EPOCHS = 45
    for i in range(EPOCHS):
        avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        print(avg_loss)

    print("finished training model")

def train_epoch(model, train_dataloader, optimizer, scheduler):
    losses = []
    for (x, y) in train_dataloader:
        optimizer.zero_grad()

        print(y)
        out = model(x)
        print(out)
        loss = (((out - y[:, 0]) / y[:, 1]) ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    return np.mean(losses)

if __name__ == "__main__":
    main()

