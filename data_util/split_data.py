#!/usr/bin/env python3

import numpy as np



def get_train_val_test_indices(datapoints):
    indices = np.arange(datapoints)
    np.random.shuffle(indices)

    train_points = int(round(datapoints * 0.7))
    val_points = int(datapoints * 0.2)
    test_points = datapoints - train_points - val_points

    train, val, test = np.split(indices, (train_points, train_points + val_points))
    assert (len(train), len(val), len(test)) == (train_points, val_points, test_points)

    return train, val, test



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


puzzles = np.memmap("./data/puzzles.dat", mode="r", dtype=np.int8, shape=(797589, 544, 8, 8))
ratings = np.memmap("./data/ratings.dat", mode="r", dtype=np.float32, shape=(797589, 2))

datapoints = 797589
train_indices, val_indices, test_indices = get_train_val_test_indices(datapoints)

create_training_partition(train_indices, puzzles, ratings, "./data/puzzles_train.dat", "./data/ratings_train.dat")
create_training_partition(val_indices, puzzles, ratings, "./data/puzzles_val.dat", "./data/ratings_val.dat")
create_training_partition(test_indices, puzzles, ratings, "./data/puzzles_test.dat", "./data/ratings_test.dat")

# dataset = PuzzleDataset(, "", "./data/shapes.csv")
