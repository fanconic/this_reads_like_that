import torch
from torchtext.datasets import AG_NEWS, IMDB
from src.models.models import MLP


def get_model(vocab_size, model_configs):
    """create a torch model with the given configs
    args:
        vocab_size: size of the vocabulary
        model_configs: dict containing the model specific parameters
    returns:
        torch model
    """
    name = model_configs["name"].lower()

    if name == "mlp":
        return MLP(vocab_size, model_configs)
    else:
        raise NotImplemented


def load_data(name):
    """Load dataset
    Args:
        name (default "MNIST"): string name of the dataset
    Returns:
        train dataset, test dataset
    """

    name = name.lower()

    if name == "ag_news":
        train_ds = AG_NEWS(split="train")
        test_ds = AG_NEWS(split="test")

    elif name == "imdb":
        train_ds = IMDB(split="train")
        test_ds = IMDB(split="test")

    else:
        raise NotImplemented

    return train_ds, test_ds
