#!/usr/bin/env bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR


download_data() {
    # reset data directory
    rm -rf data
    mkdir data

    # download puzzles
    wget https://database.lichess.org/lichess_db_puzzle.csv.bz2
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
bzip2 -kd data/lichess_db_puzzle.csv.bz2

