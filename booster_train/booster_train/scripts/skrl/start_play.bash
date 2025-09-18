#!/bin/bash
shopt -s expand_aliases
source ~/.bash_functions
_labpython play.py --task Booster-Walk-v0-Play --checkpoint $1
