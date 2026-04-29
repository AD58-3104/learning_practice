#!/bin/bash
source ~/.bash_functions
_labpython failure_model_train.py --task Parallel-failure-train-v0 --num_envs 16300  --headless agent.trainer.timesteps=400000 $@
