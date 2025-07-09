#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem 64GB
#SBATCH --gpus-per-node=v100:1

module purge
source ../distractor_preference_env/bin/activate
nvidia-smi 

python 1_retrieve_docs.py -d bio -w Simple -e Emb_Only_Options -n 60

deactivate