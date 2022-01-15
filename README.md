# This Reads Like That
This is the code of the experiments for the NLP Research Project

## Introduction
Deep neural networks, designed to provide human-interpretable decisions through their architecture, have recently become an increasingly popular alternative to the post-hoc interpretation of traditional black-box models. Perhaps the most widely used approach among these networks is the so-called prototype learning, in which similarities to learned latent prototypes serve as the basis for classifying an unseen data point. Previously, prototypical networks were mostly used in the computer vision domain with common encoder models. In this work, we research their structure to the problem of natural language processing, and more precisely to classification. We introduce a learned weighted similarity metric to extend previous work. Therefore, we compute similarity on a basis of the most informative dimensions of the sentence embedding, which outperforms traditional similarity measures in all our experiments. Additionally, we propose a way to retrieve the most similar words from the prototype and the test sample as an interpretability enhancement mechanism, which does not rely on an a priori choice of the number of words.

## Setup
### Installation
Clone this repository.
```bash
$ git clone https://github.com/fanconic/this_reads_like_that
$ cd this_reads_like_that
```

I suggest to create a virtual environment and install the required packages.
```bash
$ conda create --name nlp_env pytorch torchvision cudatoolkit=10.1 --channel pytorch
$ conda activate nlp_env
$ conda install --file requirements.txt
```

### Repository Structure
- `train.py`: Main training loop
- `rationales_training.py`: Trains the model on the human-annotated rationale dataset and computes the faithfullness
- `config.yaml`: Config yaml file, which has all the experimental settings.
- `run.sh`: Runs the training script, including the conda environment. The config.yaml file is copied to `experiment_configs`
- `gpu_experiments.sh`: Runs the training script on a SLURM cluster, using GPUs. The output file is saved in `experiment_outputs`

### Source Code Directory Tree
```
.
├── src                 # Source code            
    ├── layers              # Single Neural Network layers
    ├── models              # Neural Network Models
    ├── active              # Folder with functions for active learning
    ├── data                # Folder with data processing parts and datasets
        ├── embedding  
            ├── AG_NEWS         # precomputed embeddings of the AG NEWS dataset (needs to be downloaded and incerted from https://polybox.ethz.ch/index.php/s/S89h02V7AWDTlmw)
            ├── movies          # precomputed embeddings of the human-annotated movie reviews
            └── rt-polarity     # precomputed embeddings of the normal movie reviews
        ├── movies          # text data of the human-annotated movie reviews
        └── rt-polarity     # text data of the normal movie reviews   
    └── utils               # Useful functions, such as metrics, losses, etc
├── explanations        # Creates a CSV file with 50 test samples, and their according prototypes for interpretability
├── saved_models        # Saves the weights of the best neural networks
├── experiment_configs  # All the various configuration files for the experiments
└── experiment_outputs  # All outputs files of the experiments        
```


## How to train
On your local computer:
```
bash run.sh <your_experiment_name>
```

On a SLURM cluster:
```
bash gpu_experiments.sh <your_experiment_name>
```

## Reproduce the experiments
In order to reproduce the experiments with the ProtoTrex, you first need to download the AG_NEWS embeddings from https://polybox.ethz.ch/index.php/s/S89h02V7AWDTlmw.
Subsequently, you can rerun our experiments with the following commands:

### Weighted Similarity Experiments
- AG News - Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/cosine_news.yaml
```
- AG News - Weighted Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/weighted_cosine_news.yaml
```
- AG News - L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/l2_news.yaml
```
- AG News - Weighted L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/weighted_l2_news.yaml
```
- Movie Reviews - Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/cosine_movies.yaml
```
- Movie Reviews - Weighted Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/weighted_cosine_movies.yaml
```
- Movie Reviews - L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/l2_movies.yaml
```
- Movie Reviews - Weighted L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/weighted_L2_movies.yaml
```

### Ablation Studies of Loss
- Full Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/full_loss.yaml
```
- Without Clustering Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/remove_clust_loss.yaml
```
- Without Separation Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/remove_sep_loss.yaml
```
- Without Distribution Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/remove_distr_loss.yaml
```
- Without Diversity Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/remove_divers_loss.yaml
```
- Without L1 Regularizer: 
```bash 
python3 -u train.py --config_path experiment_configs/remove_l1_loss.yaml
```

### Faithfulness
- Train on human-annotated rationale movie review dataset 
```bash
python3 -u rationales_training.py --config experiment_configs/rationales.yaml
```

## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Severin Husmann - shusmann@ethz.ch
- Moritz Vandenhirtz - mvandenhi@ethz.ch
