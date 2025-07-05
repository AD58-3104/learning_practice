#!/bin/bash
shopt -s expand_aliases
source ~/.bash_functions
_labpython train.py --task "Booster-Kick-Play"  --checkpoint $@