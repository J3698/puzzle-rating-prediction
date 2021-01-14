from torch.utils.data import Dataset, DataLoader
from math import inf
import torch
import numpy as np

class PuzzleDataset(Dataset):
    def __init__(self, puzzles_file, ratings_file, length, truncation = inf):
        self.puzzles = np.memmap(puzzles_file, mode="r", dtype=np.int8, shape=(length, 544, 8, 8))
        self.ratings = np.memmap(ratings_file, mode="r", dtype=np.float32, shape=(length, 2))
        self.truncation = truncation

    def __len__(self):
        return min(len(self.puzzles), self.truncation)

    def __getitem__(self, idx):
        return torch.tensor(self.puzzles[idx], dtype=torch.float32),\
               torch.tensor(self.ratings[idx], dtype=torch.float32)

def get_datasets(truncation = inf):
    train_dataset = PuzzleDataset("./data/puzzles_train.dat", "./data/ratings_train.dat",
                                  558312, truncation = truncation)
    val_dataset = PuzzleDataset("./data/puzzles_val.dat", "./data/ratings_val.dat",
                                159517, truncation = truncation)
    test_dataset = PuzzleDataset("./data/puzzles_test.dat", "./data/ratings_test.dat",
                                 79760, truncation = truncation)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size, truncation = inf):
    train_dataset, val_dataset, test_dataset = get_datasets(truncation = truncation)
    print(f"Loaded datasets: {len(train_dataset)} train, "
           f"{len(val_dataset)} val, {len(test_dataset)} test")

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader, test_dataloader
