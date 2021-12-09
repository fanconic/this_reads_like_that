import torch
from torchtext.datasets import AG_NEWS, IMDB
from src.models.models import MLP, GPT2, BERT, Proto_BERT, nes_torch
import os
import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import torch.nn.functional as F

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
    elif name == "bert_baseline":
        return BERT(vocab_size, model_configs)
    elif name == "bert":
        return Proto_BERT(vocab_size, model_configs)
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
        train_ds = get_reviews(
            data_dir=kwargs["data_dir"], data_name=kwargs["data_name"], split="train"
        )
        val_ds = get_reviews(
            data_dir=kwargs["data_dir"], data_name=kwargs["data_name"], split="val"
        )
        test_ds = get_reviews(
            data_dir=kwargs["data_dir"], data_name=kwargs["data_name"], split="test"
        )

    else:
        raise NotImplemented
    return train_ds, test_ds


def get_reviews(data_dir, data_name, split="train"):
    """import the rotten tomatoes movie review dataset
    Args:
        data_dir (str): path to directory containing the data files
        data_name (str): name of the data files
        split (str "train"): data split
    Returns:
        features and labels
    """
    assert split in [
        "train",
        "val",
        "test",
    ], "Split not valid, has to be 'train', 'val', or 'test'"
    split = "dev" if split == "val" else split

    text, labels = [], []

    set_dir = os.path.join(data_dir, data_name, split)
    text_tmp = pickle.load(open(os.path.join(set_dir, "word_sequences") + ".pkl", "rb"))
    # join tokenized sentences back to full sentences for sentenceBert
    text_tmp = [detok.detokenize(sub_list) for sub_list in text_tmp]
    text.append(text_tmp)
    label_tmp = pickle.load(open(os.path.join(set_dir, "labels") + ".pkl", "rb"))
    # convert 'pos' & 'neg' to 1 & 0
    label_tmp = convert_label(label_tmp)
    labels.append(label_tmp)
    return list(zip(labels[0], text[0]))


def convert_label(labels):
    """Convert str labels into integers.
    Args:
        labels (Sequence): list of labels
    returns
        converted labels with integer mapping
    """
    converted_labels = []
    for i, label in enumerate(labels):
        if label == "pos":
            # it will be subtracted by 1 in hte label pipeline
            converted_labels.append(2)
        elif label == "neg":
            converted_labels.append(1)
    return converted_labels


def proto_loss(prototype_distances, label, model, config, device):
    # proxy variable, could be any high value
    max_dist = torch.prod(torch.tensor(model.protolayer.size()))

    # prototypes_of_correct_class is tensor of shape  batch_size * num_prototypes
    # calculate cluster cost, high cost if same class protos are far away
    # use max_dist because similarity can be >0/<0 -> shift it s.t. it's always >0
    # -> other class has value 0 which is always smaller than shifted similarity
    prototypes_of_correct_class = torch.t(
        config["model"]["prototype class"][:, label]
    ).to(device)
    inverted_distances, _ = torch.max(
        (max_dist - prototype_distances) * prototypes_of_correct_class, dim=1
    )
    clust_loss = torch.mean(max_dist - inverted_distances)
    # assures that each sample is not too far distant from a prototype of its class
    # MV: Wrong! Clust_loss does that, while distr_loss says for each prototype there is not too far sample of class
    inverted_distances, _ = torch.max(
        (max_dist - prototype_distances) * prototypes_of_correct_class, dim=0
    )
    distr_loss = torch.mean(max_dist - inverted_distances)

    # calculate separation cost, low (highly negative) cost if other class protos are far distant
    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    inverted_distances_to_nontarget_prototypes, _ = torch.max(
        (max_dist - prototype_distances) * prototypes_of_wrong_class, dim=1
    )
    sep_loss = -torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

    # diversity loss, assures that prototypes are not too close
    comb = torch.combinations(torch.arange(0, config["model"]["n_prototypes"]), r=2)
    if config["model"]["similaritymeasure"] == "cosine":
        divers_loss = torch.mean(
            F.cosine_similarity(
                model.protolayer[:, comb][:, :, 0], model.protolayer[:, comb][:, :, 1]
            )
            .squeeze()
            .clamp(min=0.8)
        )
    elif config["model"]["similaritymeasure"] == 'L2':
       divers_loss = torch.mean(nes_torch(model.protolayer[:, comb][:, :, 0],
                                          model.protolayer[:, comb][:, :, 1], dim=2).squeeze().clamp(min=0.8))

    # if args.soft:
    #    soft_loss = - torch.mean(F.cosine_similarity(model.protolayer[:, args.soft[1]], args.soft[4].squeeze(0),
    #                                                 dim=1).squeeze().clamp(max=args.soft[3]))
    # else:
    #    soft_loss = 0
    # divers_loss += soft_loss * 0.5

    # l1 loss on classification layer weights, scaled by number of prototypes
    l1_loss = model.fc.weight.norm(p=1) / config["model"]["n_prototypes"]

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss
