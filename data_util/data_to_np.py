#!/usr/bin/env python3

import numpy as np
import os
import csv
from chess import Board, Move
from fen_to_vec import fenToVec
from itertools import takewhile, repeat

DATA_FILE = "data/lichess_db_puzzle.csv"
MAX_LEN = 15 # pad input to 17 * (2 * MAX_LEN + 2)

def main():
    assert_data_downloaded()
    all_puzzles_to_data()


# count puzzles, https://stackoverflow.com/a/27518377/4142985
def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )


def all_puzzles_to_data():
    # get mmap file
    num_puzzles = rawincount(DATA_FILE)
    puzzles_file, ratings_file = get_mmap_files(num_puzzles)


    # store mmap shapes
    with open("data/shapes.csv", 'w') as shapesfile:
        writer = csv.writer(shapesfile)
        writer.writerow(["data/puzzles.dat", str(puzzles_file.shape)])
        writer.writerow(["data/puzzles.dat", str(ratings_file.shape)])


    # write to mmap
    with open(DATA_FILE) as csvfile:
        reader = csv.reader(csvfile)
        for i, (vec, rating) in enumerate(filter(lambda x: x, map(puzzle_to_data, reader))):
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

    # skip super long puzzles :)
    if (moves.count(" ") - 1) / 2 > MAX_LEN:
        return None

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
