#!/bin/bash
shopt -s expand_aliases
source ~/.bash_functions
_labpython train.py --headless --task "Booster-Kick"  $@