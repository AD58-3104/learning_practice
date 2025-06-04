#!/usr/bin/env bash

install_isaaclab_extension() {
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        uv pip install --editable $1
    fi
}

export ISAACLAB_PATH="../IsaacLab" # これはスクリプトがある場所からIsaacLabのリポジトリへの相対パス
export -f install_isaaclab_extension


find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
