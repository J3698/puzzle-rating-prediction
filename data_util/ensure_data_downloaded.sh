#!/usr/bin/env bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR


download_data() {
    # reset data directory
    rm -rf data
    mkdir data

    # download puzzles
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G3d6nVyJjvdc7sv2ZCyD4gUmwnbVyF_d' -O lichess_db_puzzle.csv.bz2
    # wget https://database.lichess.org/lichess_db_puzzle.csv.bz2
    mv lichess_db_puzzle.csv.bz2 data/
}


# parse program arguments
while [ -n "$1" ]; do # while loop starts

	case "$1" in
	-d) download_data ;; # -d to force data redownload
	*) echo "Option $1 not recognized" ;;

	esac
	shift
done


# download data if it isn't already
if ! [ -e data/lichess_db_puzzle.csv.bz2 ] ;
then
    download_data
fi


# decompress puzzles
rm data/lichess_db_puzzle.csv
bzip2 -kd data/lichess_db_puzzle.csv.bz2

