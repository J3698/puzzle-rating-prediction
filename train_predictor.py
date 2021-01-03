#!/usr/bin/env python3

# imports
import numpy as np


def main():
    print("finished training model")

    puzzles = np.memmap("./data/puzzles.dat", mode="r", dtype=np.int8, shape=(797589, 544, 8, 8))
    ratings = np.memmap("./data/ratings.dat", mode="r", dtype=np.float32, shape=(797589, 2))
    breakpoint()
    print(puzzles[0], ratings[0])



if __name__ == "__main__":
    main()
