#!/usr/bin/env bash


set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR


./data_util/ensure_data_downloaded.sh
./data_util/data_to_np.py
