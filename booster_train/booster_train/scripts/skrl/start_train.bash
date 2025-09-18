#!/bin/bash
shopt -s expand_aliases
source ~/.bash_functions
_labpython train.py --task Booster-Walk-v0 --headless $@
