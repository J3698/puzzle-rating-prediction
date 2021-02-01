#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import csv
from chess import Board, Move
from fen_to_vec import fenToVec
import matplotlib.pyplot as plt
from itertools import takewhile, repeat

DATA_FILE = "data/raw_data.csv"
MAX_PUZZLE_LEN = 15 # pad input to 17 * (2 * MAX_LEN + 2)


def main():
    assert_data_downloaded()
    all_puzzles_to_data()


def read_data():
    names = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", \
             "Popularity", "NbPlays", "Themes", "GameUrl"]
    return pd.read_csv(DATA_FILE, names = names)


def all_puzzles_to_data():
    # get mmap file
    data = read_data()
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


if __name__ == "__main__":
    main()
