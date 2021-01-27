#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import csv
from chess import Board, Move
from data_util.fen_to_vec import fenToVec
import matplotlib.pyplot as plt
from itertools import takewhile, repeat

DATA_FILE = "data/lichess_db_puzzle.csv"
MAX_LEN = 15 # pad input to 17 * (2 * MAX_LEN + 2)

def main():
    assert_data_downloaded()
    data = read_data()

    print(len(data))
    total = len(data)
    #less_than_10 =
    x = [5 * i for i in range(5)]
    histogramRatings(data)
    print(less_than_10)
    #hist_cols = ["NbPlays", "RatingDeviation"]
    #h = data[data["NbPlays"] > 20].hist(column = hist_cols, bins = 100)
    #print(len(data[data["NbPlays"] > 20]))
    #plt.show()

def histogramRatings(data):
    data.hist(column = "Rating", bins = 30)
    plt.show()

def scatterPlaysAndDeviationSmallPlays(data):
    data[data["NbPlays"] < 50].plot.scatter("NbPlays", "RatingDeviation")
    plt.show()

def scatterPlaysAndDeviation2xCleaned(data):
    data[(data["NbPlays"] < 50000) & (data["NbPlays"] >= 10)].plot.scatter("NbPlays", "RatingDeviation")
    plt.show()

def scatterPlaysAndDeviationCleaned(data):
    data[data["NbPlays"] < 50000].plot.scatter("NbPlays", "RatingDeviation")
    plt.show()

def scatterPlaysAndDeviation(data):
    data.plot.scatter("NbPlays", "RatingDeviation")
    plt.show()

def plotDeviationsLessThan10Plays(data):
    (data[data["NbPlays"] <= 10]).hist(column = "RatingDeviation", bins = 20)
    print((data["NbPlays"] <= 10).astype(bool).sum())
    plt.show()


def plotLessThan40Plays(data):
    (data[data["NbPlays"] <= 40]).hist(column = "NbPlays", bins = 20)
    print((data["NbPlays"] <= 40).astype(bool).sum())
    plt.show()


def read_data():
    names = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", \
             "Popularity", "NbPlays", "Themes", "GameUrl"]
    return pd.read_csv(DATA_FILE, names = names)


def assert_data_downloaded():
    if not os.path.exists(DATA_FILE):
        raise Exception("Data not found. Make sure you have run "
                        "ensure_data_downloaded.sh first from "
                        "the project root directory, and then this "
                        "file from the project root directory")


if __name__ == "__main__":
    main()
