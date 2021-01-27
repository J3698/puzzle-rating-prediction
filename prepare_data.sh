#!/usr/bin/env bash


# error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR


# download data if it doesn't exist
if ! [ -e data/lichess_db_puzzle.csv ] ; then
    # reset data directory
    rm -rf data
    mkdir data

    # download puzzles
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G3d6nVyJjvdc7sv2ZCyD4gUmwnbVyF_d' -O data/lichess_db_puzzle.csv.bz2

    # uncompress puzzles (deletes compressed version)
    bzip2 -d data/lichess_db_puzzle.csv.bz2
fi


# convert to vectors
./data_util/data_to_np.py


# create train/validate/test split
./data_util/split_data.py
