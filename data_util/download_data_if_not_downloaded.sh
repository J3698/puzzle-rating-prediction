#!/usr/bin/env bash


# error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR


# download data if it doesn't exist
if ! [ -e data/raw_data.csv ] ; then
    # reset data directory
    rm -rf data
    mkdir data

    # download puzzles
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G3d6nVyJjvdc7sv2ZCyD4gUmwnbVyF_d' -O data/raw_data.csv.bz2

    # uncompress puzzles (deletes compressed version)
    bzip2 -d data/raw_data.csv.bz2
fi


