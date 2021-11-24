# This Reads Like That
This is the code of the experiments for the NLP Research Project

## Introduction
TBC

## Setup
### Installation
Clone this repository.
```bash
$ git clone https://github.com/fanconic/this_reads_like_that
$ cd this_reads_like_that
```

I suggest to create a virtual environment and install the required packages.
```bash
$ conda create --name experiment_env pytorch torchvision cudatoolkit=10.1 --channel pytorch
$ conda activate experiment_env
$ conda install --file requirements.txt
```

### Repository Structure
- `run.sh`: Runs the training script, include conda environment
- `train.py`: Main training loop
- `evaluate.py`: evaluation
- `config.yaml`: Config yaml file, which has all the experimental settings.


### Source Code Directory Tree
```
.
└── src                 # Source code            
    ├── layers              # Single Neural Network layers
    ├── models              # Neural Network Models
    ├── active              # Folder with functions for active learning
    ├── data                # Folder with data processing parts
    └── utils               # Useful functions, such as metrics, losses, etc
├── model_weights       # Weights of the neural networks
├── experiment_configs  # All the various configuration files for the experiments
└── experiment_outputs  # All outputs files of the experiments        
```


## How to train
```
bash run.sh <your_experiment_name>
```

## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Severin Husmann - shusmann@ethz.ch
- Moritz Vandenhirtz - mvandenhi@ethz.ch