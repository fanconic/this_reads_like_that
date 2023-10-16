# This Reads Like That
This is the code for the EMNLP 2023 publication: __*This* Reads Like *That*: Deep Learning for Interpretable Natural Language Processing__

## Introduction
Prototype learning, a popular machine learning method designed for inherently interpretable decisions, leverages similarities to learned prototypes for classifying new data. While it is mainly applied in computer vision, in this work, we build upon prior research and further explore the extension of prototypical networks to natural language processing. We introduce a learned weighted similarity measure that enhances the similarity computation by focusing on informative dimensions of pre-trained sentence embeddings. Additionally, we propose a post-hoc explainability mechanism that extracts prediction-relevant words from both the prototype and input sentences. Finally, we empirically demonstrate that our proposed method not only improves predictive performance on the AG News and RT Polarity datasets over a previous prototype-based approach by  Friedrich et al.(2022)  but also enhances the faithfulness of explanations compared to rational-based work by Lei et al. (2016).

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
In order to reproduce the experiments with the ProtoTrex, you can download the embeddings from the various models and datasets from here [https://polybox.ethz.ch/index.php/s/S89h02V7AWDTlmw](https://polybox.ethz.ch/index.php/s/8iTOzaomCTsS1Px).
Subsequently, you can rerun our experiments with the following commands, where `<model>` should be changed with the backbone transformer (`bert`, `gpt2`, `mpnet`, `roberta`):

### Weighted Similarity Experiments
- AG News - Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/cosine_news.yaml
```
- AG News - Weighted Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/weighted_cosine_news.yaml
```
- AG News - L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/l2_news.yaml
```
- AG News - Weighted L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/weighted_l2_news.yaml
```
- Movie Reviews - Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/cosine_movies.yaml
```
- Movie Reviews - Weighted Cosine Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/weighted_cosine_movies.yaml
```
- Movie Reviews - L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/l2_movies.yaml
```
- Movie Reviews - Weighted L2 Similarity: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/weighted_L2_movies.yaml
```

### Interpretability vs. Performance Trade-Off
- AG News - non-interpretable: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/non_interpretable_news.yaml
```
- Movie Reviews - non-interpretable: 
```bash 
python3 -u train.py --config_path experiment_configs/<model>/non_interpretable_movies.yaml
```



### Ablation Studies of Loss
- Full Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/full_loss.yaml
```
- Without Clustering Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/remove_clust_loss.yaml
```
- Without Separation Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/remove_sep_loss.yaml
```
- Without Distribution Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/remove_distr_loss.yaml
```
- Without Diversity Loss: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/remove_divers_loss.yaml
```
- Without L1 Regularizer: 
```bash 
python3 -u train.py --config_path experiment_configs/bert/remove_l1_loss.yaml
```

### Faithfulness
- Train on human-annotated rationale movie review dataset 
```bash
python3 -u rationales_training.py --config experiment_configs/<model>/rationales.yaml
```

## Contributors
- Claudio Fanconi - fanconic@ethz.ch
- Moritz Vandenhirtz - moritz.vandenhirtz@inf.ethz.ch
- Severin Husmann - shusmann@ethz.ch
- Julia Vogt - julia.vogt@inf.ethz.ch
