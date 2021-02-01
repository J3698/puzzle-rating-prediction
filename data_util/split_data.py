#!/usr/bin/env python3

import numpy as np


def main():
    data/lichess_db_puzzle.csv
    puzzles = np.memmap("./data/puzzles.dat", mode="r", dtype=np.int8, shape=(797589, 544, 8, 8))
    ratings = np.memmap("./data/ratings.dat", mode="r", dtype=np.float32, shape=(797589, 2))

    datapoints = 797589
    train_indices, val_indices, test_indices = get_train_val_test_indices(datapoints)
    create_dataset_partition(train_indices, puzzles, ratings, "./data/puzzles_train.dat", "./data/ratings_train.dat")
    create_dataset_partition(val_indices, puzzles, ratings, "./data/puzzles_val.dat", "./data/ratings_val.dat")
    create_dataset_partition(test_indices, puzzles, ratings, "./data/puzzles_test.dat", "./data/ratings_test.dat")


def get_train_val_test_indices(datapoints, train_ratio, val_ration, test_ratio):
    assert 1 == train_ratio + val_ratio + test_ratio, "train/val/test ratios must add to 1"

    # get shuffled indices
    np.random.seed(0)
    indices = np.arange(datapoints)
    np.random.shuffle(indices)

    # calculate number of datapoints in each set
    num_train_points = int(round(datapoints * 0.7))
    num_val_points = int(datapoints * 0.2)
    num_test_points = datapoints - train_points - val_points

    # split indices
    split_locations = (num_train_points, num_train_points + num_val_points)
    train_indices, val_indices, test_indices = np.split(indices, split_locations)
    assert (len(train_indices), len(val_indices), len(test_indices)) == (train_points, val_points, test_points)

    print(f"Train points: {len(train)}, Val points: {len(val)}, Test points: {len(test)}")
    return train_indices, val_indices, test_indices


def create_training_partition(indices, data_x, data_y, filename_x, filename_y):
    puzzle_shape = (len(indices), 544, 8, 8)
    rating_shape = (len(indices), 2)

    puzzles_file = np.memmap(filename_x, dtype = np.int8, mode = 'write', shape = puzzle_shape)
    ratings_file = np.memmap(filename_y, dtype = np.float32, mode = 'write', shape = rating_shape)

    for i, idx in enumerate(indices):
        puzzles_file[i] = data_x[idx]
        ratings_file[i] = data_y[idx]

    del puzzles_file
    del ratings_file



if __name__ == "__main__":
    main()
