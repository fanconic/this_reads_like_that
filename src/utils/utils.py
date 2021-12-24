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
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
import re

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
            model.parameters(),
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
    """ Calculate the various losses of the prototypes
    Args:
        prototype_distances: distances/similarities of the prototypes
        label: correct targets
        config: dict containing further configurations
        device: current pytorch device
    Returns:
        the various loss values
    """
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
    comb = torch.cartesian_prod(
        torch.arange(0, config["model"]["n_prototypes"]),
        torch.arange(0, config["model"]["n_prototypes"]),
    )
    if config["model"]["similaritymeasure"] == "cosine":

        proto_sim = (
            F.cosine_similarity(
                model.protolayer[:, comb][:, :, 0],
                model.protolayer[:, comb][:, :, 1],
                dim=2,
            )
            .squeeze()
            .reshape((config["model"]["n_prototypes"], config["model"]["n_prototypes"]))
        )
        proto_sim += torch.diag(-3 * torch.ones(config["model"]["n_prototypes"])).to(
            device
        )  # decrease self-similarity to not be max
        proto_min_sim, _ = torch.max(proto_sim, dim=1)
        divers_loss = torch.mean(proto_min_sim)
    elif config["model"]["similaritymeasure"] == "L2":
        proto_dist = torch.cdist(model.protolayer, model.protolayer, p=2) / np.sqrt(
            config["model"]["embed_dim"]
        )
        proto_dist += torch.diag(200 * torch.ones(config["model"]["n_prototypes"])).to(
            device
        )  # Increase self distance to not pick this as closest
        proto_min_dist, _ = torch.min(proto_dist, dim=1)
        assert torch.all(
            proto_min_dist < 200
        )  # Set self-distance to 200 -> Don't take that
        divers_loss = -torch.mean(proto_min_dist)
    elif config["model"]["similaritymeasure"] == "weighted cosine":
        proto_sim = ((torch.sum(model.dim_weights*model.protolayer[:, comb][:, :, 0]*model.protolayer[:, comb][:, :, 1], dim=-1)/torch.maximum((
                torch.sqrt(torch.sum(model.dim_weights*torch.square(model.protolayer[:, comb][:, :, 0]),dim=-1))*torch.sqrt(torch.sum(model.dim_weights*torch.square(model.protolayer[:, comb][:, :, 1]),dim=-1))),torch.tensor(1e-8)
            ))
            .squeeze()
            .reshape((config["model"]["n_prototypes"], config["model"]["n_prototypes"]))
        )

        proto_sim += torch.diag(-3 * torch.ones(config["model"]["n_prototypes"])).to(
            device
        )  # decrease self-similarity to not be max
        proto_min_sim, _ = torch.max(proto_sim, dim=1)
        divers_loss = torch.mean(proto_min_sim)
    else:
        print("loss not defined")
        assert False
    l1_loss = model.fc.weight.norm(p=1) / config["model"]["n_prototypes"]

    return distr_loss, clust_loss, sep_loss, divers_loss, l1_loss


def save_embedding(embedding, mask, label, config, set_name):
    """ Save the embeddings, to speed up computing
    Args:
        embedding: the sentence embeddings
        mask: according mask of the embeddings
        label: targets
        config: further configuration
        set_name: name of the phase
    """
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
    """ Load the precomputed embeddings
    Args:
        config: configuration dict
        set_name: name of the current set
    returns:
        embedding, masks & labels
    """
    path = os.path.join("./src/data/embedding", config["data"]["data_name"])
    name = config["model"]["submodel"] + "_" + set_name
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
    """ Loads the model and data loader
    Args:
        wandb: wandb instance for logging
        config: configuration file
        device: current device
    Returns:
        model and dataloaders
    """
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
        train_loader_unshuffled = torch.utils.data.DataLoader(
            list(zip(labels_train, embedding_train, mask_train)),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
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
    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        train_loader_unshuffled,
    )


def get_nearest_sent(config, model, train_loader, device):
    """ Retrueve the nearest sentence
    Args:
        config: configuration dict
        model: classification model
        train_loader: data loader with the train data
        device: current device
    Returns:
        embedding of the nearest embeddings
    """
    # Easier code which might only work for sentence
    model.eval()
    dist = []
    labels = []
    embeddings = []
    with torch.no_grad():
        for idx, (label, text, mask) in enumerate(train_loader):
            text = text.to(device)
            mask = mask.to(device)
            distances, _ = model.get_dist(text.unsqueeze(1), mask)
            embeddings.append(text)
            labels.append(label)
            prototypes_of_correct_class = torch.t(
                config["model"]["prototype class"][:, label].to(device)
            )
            # Only look at embeddings of same class
            proto_dist = (
                prototypes_of_correct_class * (distances - 100) + 100
            )  # Ugly hack s.t. distance of non-classes are 100 (=big)
            dist.append(proto_dist)
    dist = torch.cat(dist)
    values, nearest_ids = torch.topk(dist, 20, dim=0, largest=False)
    assert (
        torch.max(values) < 100
    )  # Check that hack works, otherwise datapoint from other class is closer to prototype
    nearest_ids = nearest_ids.cpu().detach().numpy().T
    for j in range(1, 20):

        for i in range(len(nearest_ids[:, 0])):
            if np.count_nonzero(nearest_ids[:, 0] == nearest_ids[i, 0]) != 1:
                nearest_ids[i, 0] = nearest_ids[i, j]
        if len(np.unique(nearest_ids[:, 0])) == len(nearest_ids[:, 0]):
            break

    assert len(np.unique(nearest_ids[:, 0])) == len(nearest_ids[:, 0])
    new_proto_emb = torch.cat(embeddings)[
        nearest_ids[:, 0], :
    ]  # model.protolayer[:,13:15] goes to nan in optimizer
    return new_proto_emb


def project(config, model, train_loader, device, last_proj):
    """
    Project the sentences back on the input space
    Args:
        config: configuration dict
        model: classification models
        train_loader: data loader with the training data
        last_proj: Last projection
    Returns:
        model
    """
    # project prototypes
    if config["model"]["embedding"] == "sentence":
        new_proto_emb = get_nearest_sent(config, model, train_loader, device)
        new_proto = new_proto_emb
    elif config["model"]["embedding"] == "word":
        print("Word projection, not yet implemented")
        raise NotImplemented
    else:
        print("Specify sentence or word in config")
    new_proto = new_proto.view(model.protolayer.shape)
    model.protolayer.copy_(new_proto)

    # Newly define Prototypes for freezing them, otherwise Adam continues updating bcs of Running Average
    if last_proj:  
        model.protolayer = nn.parameter.Parameter(new_proto, requires_grad=False,)
    # give prototypes their "true" label
    return model


def mean_pooling(model_output, attention_mask):
    """ Computes the average and pools the values
    Args:
        model_output: output features of the classification model
        attention_mask: according atttention mask
    Returns:
        pooled embeddings
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def prototype_visualization(config, model, train_ds, train_loader_unshuffled, device):
    """ Visualize the prototypical sentences, to receive an interpretable reasoning of the model
    Args:
        config: configuration dict
        model: classification model
        train_ds: training dataset
        train_loader_unshuffled: a data loader with the training data that is UNSHUFFLED
        device: current device
    """
    print("Prototype Visualization in Progress!")
    # Easier code which might only work for sentence
    model.eval()
    dist = []
    labels = []
    embeddings = []
    with torch.no_grad():
        for idx, (label, text, mask) in enumerate(train_loader_unshuffled):
            text = text.to(device)
            mask = mask.to(device)
            distances, _ = model.get_dist(text.unsqueeze(1), mask)
            embeddings.append(text)
            labels.append(label)
            prototypes_of_correct_class = torch.t(
                config["model"]["prototype class"][:, label].to(device)
            )
            # Only look at embeddings of same class
            proto_dist = (
                prototypes_of_correct_class * (distances - 100) + 100
            )  # Ugly hack s.t. distance of non-classes are 100 (=big)
            dist.append(proto_dist)
    dist = torch.cat(dist)
    nearest_vals, nearest_ids = torch.topk(dist, 1, dim=0, largest=False)
    nearest_vals = nearest_vals.cpu().detach().numpy()
    nearest_ids = nearest_ids.cpu().detach().numpy().T.squeeze()
    texts = []
    for idx, (label, text) in enumerate(train_ds):
        texts.append(text)
    prototext = [texts[i] for i in nearest_ids]
    # Find subset of words giving meaning to sentence-prototype
    if (
        config["model"]["embedding"] == "sentence"
        and config["model"]["submodel"] == "bert"
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/bert-large-nli-mean-tokens"
        )
        model_emb = AutoModel.from_pretrained(
            "sentence-transformers/bert-large-nli-mean-tokens"
        )
    # Create Variations of all Sentence Embeddings by removing one word
    keep_words = []

    for nth_proto in range(len(prototext)):
        proto_strings = prototext[nth_proto]
        top_words = np.min(
            (5, len(re.findall(r"[\w']+|[.,\":\[\]!?;]", proto_strings)))
        )
        proto_words = []
        proto_distance = np.empty(top_words)

        for nth_removed_word in range(
            top_words
        ):  # Iteratively remove most important words
            prototype = re.findall(r"[\w']+|[.,\":\[\]!?;]", proto_strings)
            sentence_variants = [
                prototype[:i] + prototype[i + 1 :] for i in range(len(prototype))
            ]
            left_word = [prototype[i] for i in range(len(prototype))]
            sentence_variants = [" ".join(i) for i in sentence_variants]
            tokenized_proto = tokenizer(
                sentence_variants, padding=True, truncation=True, return_tensors="pt"
            )
            # Compute token embeddings
            with torch.no_grad():
                model_output = model_emb(**tokenized_proto)
            # Perform pooling. In this case, mean pooling.
            sentence_embeddings = mean_pooling(
                model_output, tokenized_proto["attention_mask"]
            ).to(device)
            # Calculate distance to orginial embedding of sentence.
            dist_per_word, _ = model.get_dist(sentence_embeddings.unsqueeze(1), _)
            dist_per_word = dist_per_word[:, nth_proto]
            farthest_val, farthest_ids = torch.topk(
                dist_per_word, 1, dim=0, largest=True
            )  # Store largest distance
            proto_words.append(left_word[farthest_ids])
            proto_distance[nth_removed_word] = farthest_val
            proto_strings = sentence_variants[farthest_ids]

        # Choose words that give 75% of distance of all 5 words
        proto_word_dist = proto_distance - nearest_vals[0, nth_proto]
        cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
        cutoff[0] = True  # Always use first word
        keep_words.append([proto_words[i] for i in np.where(cutoff)[0]])
    for j in range(config["model"]["n_classes"]):
        index = np.where(config["model"]["prototype class"][:, j] == 1)[0]
        print("Class {}:".format(j))  # For RT: 0 = Bad, 1 = Good
        for i in index:
            print(np.array(keep_words[i]), sep="\n")
            print(np.array(prototext[i]), sep="\n")

