import torch
from torchtext.datasets import AG_NEWS, IMDB
from src.models.models import MLP, GPT2, BERT, ProtoNet, nes_torch
from src.data.dataloader import build_loader
import os
import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import torch.nn.functional as F
import numpy as np

from torch import nn, optim

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
    elif name == "proto":
        return ProtoNet(vocab_size, model_configs)
    else:
        raise NotImplemented


def get_optimizer(model, config):
    """Resolve the optimizer according to the configs
    Args:
        model: model on which the optimizer is applied on
        config: configuration dict
    returns:
        optimizer
    """
    if config["optimizer"]["name"] == "SGD":
        optimizer = optim.SGD(
            model.optim_parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            betas=config["optimizer"]["betas"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    return optimizer


def get_scheduler(optimizer, config):
    """Take the specified scheduler
    Args:
        optimizer: optimizer on which the scheduler is applied
        config: configuration dict
    returns:
        resovled scheduler
    """
    scheduler_name = config["scheduler"]["name"]
    assert scheduler_name in [
        "reduce_on_plateau",
        "step",
        "poly",
        "CosAnnWarmup",
    ], "scheduler not Implemented"

    if scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler"]["lr_reduce_factor"],
            patience=config["scheduler"]["patience_lr_reduce"],
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler"]["patience_lr_reduce"],
            gamma=config["scheduler"]["lr_reduce_factor"],
        )
    elif scheduler_name == "poly":
        epochs = config["train"]["epochs"]
        poly_reduce = config["scheduler"]["poly_reduce"]
        lmbda = lambda epoch: (1 - (epoch - 1) / epochs) ** poly_reduce
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)

    elif scheduler_name == "CosAnnWarmup":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["optimizer"]["T0"],
            T_mult=1,
            eta_min=config["optimizer"]["lr"] * 1e-2,
            last_epoch=-1,
        )

    else:
        scheduler = None
    return scheduler


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


def ned_torch(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)


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
    elif config["model"]["similaritymeasure"] == "L2":
        divers_loss = torch.mean(
            nes_torch(
                model.protolayer[:, comb][:, :, 0],
                model.protolayer[:, comb][:, :, 1],
                dim=2,
            )
            .squeeze()
            .clamp(min=0.8)
        )

    # if args.soft:
    #    soft_loss = - torch.mean(F.cosine_similarity(model.protolayer[:, args.soft[1]], args.soft[4].squeeze(0),
    #                                                 dim=1).squeeze().clamp(max=args.soft[3]))
    # else:
    #    soft_loss = 0
    # divers_loss += soft_loss * 0.5

    # l1 loss on classification layer weights, scaled by number of prototypes
    l1_loss = model.fc.weight.norm(p=1) / config["model"]["n_prototypes"]

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss


def save_embedding(embedding, mask, label, config, set_name):
    path = os.path.join("./src/data/embedding", config["data"]["data_name"])
    os.makedirs(path, exist_ok=True, mode=0o777)
    name = config["model"]["name"] + "_" + set_name
    path_e = os.path.join(path, name + ".pt")
    torch.save(embedding, path_e)
    path_m = os.path.join(path, name + "_mask.pt")
    torch.save(mask, path_m)
    path_l = os.path.join(path, name + "_label.pt")
    torch.save(label, path_l)


def load_embedding(config, set_name):
    path = os.path.join("./src/data/embedding", config["data"]["data_name"])
    name = config["model"]["name"] + "_" + set_name
    path_e = os.path.join(path, name + ".pt")
    assert os.path.isfile(path_e)
    path_m = os.path.join(path, name + "_mask.pt")
    assert os.path.isfile(path_m)
    path_l = os.path.join(path, name + "_label.pt")
    assert os.path.isfile(path_l)
    embedding = torch.load(path_e, map_location=torch.device("cpu"))
    mask = torch.load(path_m, map_location=torch.device("cpu"))
    label = torch.load(path_l, map_location=torch.device("cpu")).to(torch.long)
    return embedding, mask, label


def load_model_and_dataloader(wandb, config, device):
    train_iter, test_iter = load_data(
        config["data"]["dataset"],
        data_dir=config["data"]["data_dir"],
        data_name=config["data"]["data_name"],
    )
    # obtain training indices that will be used for validation
    num_train = len(train_iter)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(config["data"]["val_size"] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_iter = list(train_iter)
    test_ds = list(test_iter)

    train_ds = torch.utils.data.Subset(train_iter, train_idx)
    val_ds = torch.utils.data.Subset(train_iter, valid_idx)

    # build the tokenized vocabulary:
    if not config["model"]["embedding"] == "sentence":
        train_loader, vocab = build_loader(
            train_ds,
            device=device,
            batch_size=config["train"]["batch_size"],
            config=config,
        )
        val_loader, _ = build_loader(
            val_ds,
            vocab=vocab,
            device=device,
            batch_size=config["train"]["batch_size"],
            config=config,
        )
        test_loader, _ = build_loader(
            test_ds,
            vocab=vocab,
            device=device,
            batch_size=config["train"]["batch_size"],
            config=config,
        )

        # get the model
        model = get_model(vocab_size=len(vocab), model_configs=config["model"]).to(
            device
        )
        wandb.watch(model)
    else:  # SentBert with embeddings beforehand
        model = get_model(vocab_size=None, model_configs=config["model"]).to(device)
        wandb.watch(model)
        if not config["data"]["compute_emb"]:
            embedding_train, mask_train, labels_train = load_embedding(config, "train")
            embedding_val, mask_val, labels_val = load_embedding(config, "val")
            embedding_test, mask_test, labels_test = load_embedding(config, "test")
        else:
            embedding_train, mask_train, labels_train = model.compute_embedding(
                train_ds, config, device
            )
            embedding_val, mask_val, labels_val = model.compute_embedding(
                val_ds, config, device
            )
            embedding_test, mask_test, labels_test = model.compute_embedding(
                test_ds, config, device
            )
            save_embedding(embedding_train, mask_train, labels_train, config, "train")
            save_embedding(embedding_val, mask_val, labels_val, config, "val")
            save_embedding(embedding_test, mask_test, labels_test, config, "test")
            torch.cuda.empty_cache()  # free up language model from GPU
        train_loader = torch.utils.data.DataLoader(
            list(zip(labels_train, embedding_train, mask_train)),
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            list(zip(labels_val, embedding_val, mask_val)),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
        test_loader = torch.utils.data.DataLoader(
            list(zip(labels_test, embedding_test, mask_test)),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
    return model, train_loader, val_loader, test_loader
