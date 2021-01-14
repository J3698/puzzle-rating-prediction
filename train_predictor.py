#!/usr/bin/env python3

# imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import *

TRUNCATION = 3

class PuzzleDataset(Dataset):
    def __init__(self, puzzles_file, ratings_file, length):
        self.puzzles = np.memmap(puzzles_file, mode="r", dtype=np.int8, shape=(length, 544, 8, 8))
        self.ratings = np.memmap(ratings_file, mode="r", dtype=np.float32, shape=(length, 2))

    def __len__(self):
        return min(len(self.puzzles), TRUNCATION)

    def __getitem__(self, idx):
        return torch.tensor(self.puzzles[idx].copy(), dtype=torch.float32),\
               torch.tensor(self.ratings[idx].copy(), dtype=torch.float32)


def main():
    train_dataset = PuzzleDataset("./data/puzzles_train.dat", "./data/ratings_train.dat", 558312)
    val_dataset = PuzzleDataset("./data/puzzles_val.dat", "./data/ratings_val.dat", 159517)
    test_dataset = PuzzleDataset("./data/puzzles_test.dat", "./data/ratings_test.dat", 79760)

    print(f"Loaded datasets: {len(train_dataset)} train, "
           f"{len(val_dataset)} val, {len(test_dataset)} test")
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = True)
    model = BasicChessCNN()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.3)

    EPOCHS = 45
    for i in range(EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, scheduler)
        val_loss = evaluate(model, val_dataloader)
        print(f"TRAIN LOSS {train_loss:2f}, VAL LOSS {val_loss:2f}\n")

    print("finished training model")


def scaled_l2_loss(actual, desired):
    """
        actual  (batch_size, 1): each value is a predicated value
        desired (batch_size, 2): each row is a value and then uncertainty of that value
    """
    batch, _ = actual.shape
    assert actual.shape == (batch, 1), actual.shape
    assert desired.shape == (batch, 2), desired.shape

    std_dev = desired[:, 1, None]
    assert std_dev.shape == (batch, 1), std_dev.shape

    scaled_actual = actual / std_dev
    assert scaled_actual.shape == (batch, 1), scaled_actual.shape

    scaled_desired = desired[:, 0, None] / std_dev
    assert scaled_desired.shape == (3, 1), scaled_desired.shape
    print("sa", scaled_actual)
    print("sd", scaled_desired)

    return F.mse_loss(scaled_actual, scaled_desired)


def train(model, train_dataloader, optimizer, scheduler):
    losses = []
    for (x, y) in train_dataloader:
        optimizer.zero_grad()

        print(y)
        out = model(x)
        print(out)
        loss = scaled_l2_loss(out, y)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step(loss)

    return np.mean(losses)


def evaluate(model, eval_dataloader):
    with torch.no_grad():
        losses = []
        for (x, y) in eval_dataloader:
            out = model(x)
            loss = scaled_l2_loss(out, y)
            losses.append(loss.item())

    return np.mean(losses)

if __name__ == "__main__":
    main()

