#!/bin/bash
source /itet-stor/fanconic/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu10

cp config.yaml experiment_configs/$1.yaml
python3 -u train.py --config_path experiment_configs/$1.yaml