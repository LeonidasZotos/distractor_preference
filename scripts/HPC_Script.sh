#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem 64GB
#SBATCH --gpus-per-node=v100:2

module purge
source ../distractor_preference_env/bin/activate
nvidia-smi 

python 1_retrieve_docs.py

deactivate