#!/usr/bin/env bash

# error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' ERR

# delegate to other scripts
./data_util/ensure_data_downloaded.sh
./data_util/data_to_np.py
./data_util/split_data.py
