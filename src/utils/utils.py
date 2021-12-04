import torch
from torchtext.datasets import AG_NEWS, IMDB
from src.models.models import MLP, GPT2, BERT
import os
import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

tok = TreebankWordTokenizer()
detok = TreebankWordDetokenizer()


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
    elif name == "gpt2":
        return GPT2(vocab_size, model_configs)
    elif name == "bert":
        return BERT(vocab_size, model_configs)
    else:
        raise NotImplemented


def load_data(name, **kwargs):
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

    elif name == "reviews":
        train_ds = get_reviews(data_dir=kwargs["data_dir"],
                               data_name=kwargs["data_name"], split="train")
        val_ds = get_reviews(data_dir=kwargs["data_dir"],
                             data_name=kwargs["data_name"], split="val")
        test_ds = get_reviews(data_dir=kwargs["data_dir"],
                              data_name=kwargs["data_name"], split="test")

    else:
        raise NotImplemented
    return train_ds, test_ds


def get_reviews(data_dir, data_name, split="train"):
    """ import the rotten tomatoes movie review dataset
    Args:
        data_dir (str): path to directory containing the data files
        data_name (str): name of the data files
        split (str "train"): data split
    Returns:
        features and labels
    """
    assert split in [
        'train', 'val', 'test'], "Split not valid, has to be 'train', 'val', or 'test'"
    split = "dev" if split == "val" else split

    text, labels = [], []

    set_dir = os.path.join(data_dir, data_name, split)
    text_tmp = pickle.load(
        open(os.path.join(set_dir, 'word_sequences') + '.pkl', 'rb'))
    # join tokenized sentences back to full sentences for sentenceBert
    text_tmp = [detok.detokenize(sub_list) for sub_list in text_tmp]
    text.append(text_tmp)
    label_tmp = pickle.load(
        open(os.path.join(set_dir, 'labels') + '.pkl', 'rb'))
    # convert 'pos' & 'neg' to 1 & 0
    label_tmp = convert_label(label_tmp)
    labels.append(label_tmp)
    return list(zip(labels[0], text[0]))


def convert_label(labels):
    """ Convert str labels into integers.
    Args:
        labels (Sequence): list of labels
    returns
        converted labels with integer mapping
    """
    converted_labels = []
    for i, label in enumerate(labels):
        if label == 'pos':
            # it will be subtracted by 1 in hte label pipeline
            converted_labels.append(2)
        elif label == 'neg':
            converted_labels.append(1)
    return converted_labels
