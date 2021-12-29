#!/bin/bash
cp config.yaml experiment_configs/$1.yaml
python3 -u train.py --config_path experiment_configs/$1.yaml