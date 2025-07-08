#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --mem 32GB
#SBATCH --gpus-per-node=v100:1

module purge
source ../distractor_preference_env/bin/activate
nvidia-smi 

python 1_retrieve_docs.py

deactivate