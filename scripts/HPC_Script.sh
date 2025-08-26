#!/bin/bash

#SBATCH --time=01:30:00
#SBATCH --mem 64GB
#SBATCH --gpus-per-node=a100:1

module purge
source ../distractor_preference_env/bin/activate
nvidia-smi 

python 1_retrieve_docs.py -d usmle -w En -e Emb -n 20
python 1_retrieve_docs.py -d usmle -w En -e Emb -n 60


deactivate