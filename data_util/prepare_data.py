
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import csv
from chess import Board, Move
from fen_to_vec import fenToVec
import matplotlib.pyplot as plt
from itertools import takewhile, repeat

DATA_HEADERS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", \
                "Popularity", "NbPlays", "Themes", "GameUrl"]

RAW_DATA_FILENAME = "data/raw_data.csv"
FILTERED_DATA_FILENAME = "data/filtered_data.csv"

DATA_SPLIT_NAMES = ("data/train", "data/val", "data/test")
DATA_SPLIT_RATIOS = (0.7, 0.2, 0.1)

MAX_PUZZLE_LEN = 15 # pad input to 17 * (2 * MAX_LEN + 2)


def main():
    assert_data_downloaded()
    print("Found data")

    create_filtered_csv(RAW_DATA_FILENAME, FILTERED_DATA_FILENAME)
    print(f"Filtered data into {FILTERED_DATA_FILENAME}")

    randomly_partition_csv(FILTERED_DATA_FILENAME, DATA_SPLIT_NAMES, DATA_SPLIT_RATIOS)
    print(f"Dataset splits {DATA_SPLIT_NAMES} created with ratios {DATA_SPLIT_RATIOS}.")



    all_puzzles_to_data()

    puzzles = np.memmap("./data/puzzles.dat", mode="r", dtype=np.int8, shape=(797589, 544, 8, 8))
    ratings = np.memmap("./data/ratings.dat", mode="r", dtype=np.float32, shape=(797589, 2))

    datapoints = 797589
    train_indices, val_indices, test_indices = get_train_val_test_indices(datapoints)
    create_dataset_partition(train_indices, puzzles, ratings, "./data/puzzles_train.dat", "./data/ratings_train.dat")
    create_dataset_partition(val_indices, puzzles, ratings, "./data/puzzles_val.dat", "./data/ratings_val.dat")
    create_dataset_partition(test_indices, puzzles, ratings, "./data/puzzles_test.dat", "./data/ratings_test.dat")


def create_filtered_csv(raw_filename, output_filename):
    raw_data_df = read_raw_data(raw_filename)

    filtered_data_df = high_rating_deviation_puzzles_removed(raw_data_df, 300)
    filtered_data_df = rarely_played_puzzles_removed(filtered_data_df, 10)

    filtered_data_df.to_csv(output_filename)


def high_rating_deviation_puzzles_removed(data_df, max_deviation_allowed):
    return data_df[data_df["RatingDeviation"] <= max_deviation_allowed]


def rarely_played_puzzles_removed(data_df, min_plays_allowed):
    return data_df[data_df["NbPlays"] >= min_plays_allowed]


def read_raw_data():
    return pd.read_csv(DATA_FILE, names = DATA_HEADERS)


def all_puzzles_to_data():
    # get mmap file
    data = read_raw_data()
    hist_cols = ["NbPlays", "RatingDeviation"]
    h = data[data["NbPlays"] > 20].hist(column = hist_cols, bins = 100)
    print(len(data[data["NbPlays"] > 20]))
    plt.show()
    data = remove_puzzles_with_large_uncertainty(data)
    num_puzzles = count_puzzles_with_valid_deviation(reader)
    puzzles_file, ratings_file = get_mmap_files(num_puzzles)


    # store mmap shapes
    with open("data/shapes.csv", 'w') as shapesfile:
        writer = csv.writer(shapesfile)
        writer.writerow(["data/puzzles.dat", str(puzzles_file.shape)])
        writer.writerow(["data/puzzles.dat", str(ratings_file.shape)])


    # write to mmap
    with open(DATA_FILE) as csvfile:
        for i, (vec, rating) in enumerate([puzzle_to_data(i) for i in reader if i]):
            puzzles_file[i] = vec
            ratings_file[i] = rating


    # save mmap
    del puzzles_file, ratings_file


def get_mmap_files(num_puzzles):
    puzzle_shape = (num_puzzles, 17 * (2 * MAX_LEN + 2), 8, 8)
    rating_shape = (num_puzzles, 2)
    puzzles_file = np.memmap("data/puzzles.dat", dtype = np.int8, mode = 'write', shape = puzzle_shape)
    ratings_file = np.memmap("data/ratings.dat", dtype = np.float32, mode = 'write', shape = rating_shape)
    return puzzles_file, ratings_file


def puzzle_to_data(puzzle):
    _, fen, moves, rating, rating_deviation, _, plays, tags, _ = puzzle

    # we don't like super long puzzles :)
    if (moves.count(" ") - 1) / 2 > MAX_LEN:
        raise Exception("Puzzle too long")

    # get puzzle start board vector
    board = Board(fen)
    vecs = [fenToVec(board.fen())]

    # add subsequent board vectors
    for move in map(Move.from_uci, moves.split(" ")):
        board.push(move)
        vec = fenToVec(board.fen()).astype(np.int8)
        assert vec.shape == (17, 8, 8)
        vecs.append(vec)

    # pad so each puzzle is same shape
    vecs = np.concatenate(vecs, axis = 0)
    assert 17 * (2 * MAX_LEN + 2) - vecs.shape[0] >= 0 # 2x check that we threw away long puzzles
    vecs = np.pad(vecs, ((0, 17 * (2 * MAX_LEN + 2) - vecs.shape[0]), (0, 0), (0, 0)))

    return vecs, np.array([float(rating), float(rating_deviation)])


def assert_data_downloaded():
    if not os.path.exists(DATA_FILE):
        raise Exception("Data not found. Make sure you have run "
                        "ensure_data_downloaded.sh first from "
                        "the project root directory, and then this "
                        "file from the project root directory")


def randomly_partition_csv(filename, partition_filenames, partition_ratios):
    data_df = pandas.read_csv(filename)

    partitioning_indices = get_partitioning_indices(len(data_df), partition_ratios)
    indices_and_filenames = zip(partitioning_indices, partition_filenames)
    for partition, filename in indices_and_filenames:
        data_df.take(partition).to_csv(filename + ".csv")


def get_partitioning_indices(num_datapoints, partition_ratios):
    assert sum(partition_ratios) == 1, "partition ratios must add to 1"

    # get shuffled indices
    np.random.seed(0)
    indices = np.arange(datapoints)
    np.random.shuffle(indices)

    # calculate number of datapoints in each set
    points_per_partition_list = [int(round(i * num_datapoints)) for i in partition_ratios]
    assert sum(points_per_partition_list) == num_datapoints, "Partitioned points don't span dataset"

    # split indices
    split_locations = np.cumsum(points_per_partition_list[:-1])
    partitioning_indices = np.split(indices, split_locations)
    assert all(len(partitioning_indices[j]) == points_per_partition_list[j] for j in range(len(points_per_partition_list)))

    return partitioning_indices


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
