# Taken from this tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import os
import torch
from torch import nn, optim
import yaml
import wandb
import argparse
import pandas as pd
import random
import os
import numpy as np
from src.utils.utils import load_data, get_model
from src.data.dataloader import build_loader
import time
import IPython
from tqdm import tqdm


def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(config, random_state=0):
    use_cuda = torch.cuda.is_available()
    print("Cuda is available: ", use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(random_state)

    train_iter, test_iter = load_data(config["data"]["dataset"])
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
    train_loader, vocab = build_loader(
        train_ds, device=device, batch_size=config["train"]["batch_size"], config=config)
    val_loader, _ = build_loader(
        val_ds, vocab=vocab, device=device, batch_size=config["train"]["batch_size"], config=config)
    test_loader, _ = build_loader(
        test_ds, vocab=vocab, device=device, batch_size=config["train"]["batch_size"], config=config)

    verbose = config["train"]["verbose"]

    # get the model
    model = get_model(vocab_size=len(vocab), model_configs=config["model"])

    wandb.watch(model)

    # prepare teh optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=config["optimizer"]["betas"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training loop
    model.train()
    total_acc, total_count = 0, 0
    val_total_acc, val_total_count = 0, 0

    epochs = config["train"]["epochs"]
    gpt2_bert_lm = config["model"]["name"] in ['gpt2', 'bert']

    for epoch in range(epochs):
        if verbose:
            train_loader = tqdm(train_loader)
            val_loader = tqdm(val_loader)
            test_loader = tqdm(test_loader)

        # Training Loop
        model.train()
        for idx, (label, text, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(text, mask)
            predicted_label = predicted_label.logits if gpt2_bert_lm else predicted_label
            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

            # calculate metric
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if verbose:
                train_loader.set_description(f"Epoch [{epoch}/{epochs}]")
                train_loader.set_postfix(
                    loss=loss.item(), acc=total_acc / total_count)
            wandb.log({
                "train_loss": loss,
                "train_accuracy": total_acc / total_count})

        model.eval()
        # Validation Loop
        for idx, (label, text, mask) in enumerate(val_loader):
            predicted_label = model(text, mask)
            predicted_label = predicted_label.logits if gpt2_bert_lm else predicted_label
            val_loss = criterion(predicted_label, label)
            val_total_acc += (predicted_label.argmax(1) == label).sum().item()
            val_total_count += label.size(0)
            if verbose:
                val_loader.set_description(f"Epoch [{epoch}/{epochs}]")
                val_loader.set_postfix(
                    loss=val_loss.item(), acc=val_total_acc / val_total_count)

        # end of epoch
        print('| epoch {:3d} | accuracy {:8.3f} | validation accuracy {:8.3f}'.format(
            epoch, total_acc / total_count, val_total_acc / val_total_count))
        wandb.log({"epoch": epoch,
                   "val_loss": val_loss,
                   "val_accuracy": val_total_acc / val_total_count})

        total_acc, total_count = 0, 0
        val_total_acc, val_total_count = 0, 0

    # Test the model
    model.eval()
    total_acc, total_count = 0, 0

    test_losses = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(test_loader):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.logits if gpt2_bert_lm else predicted_label
            test_loss = criterion(predicted_label, label)
            test_losses.append(test_loss)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

        wandb.log({"test_loss": np.mean(test_losses),
                   "test_accuracy": total_acc/total_count
                   })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    print("Running experiment: {}".format(config["name"]))

    # Weights & Biases for tracking training
    wandb.init(
        mode="disabled",
        project="nlp_groupproject",
        entity="nlp_groupproject",
        name=config["name"],
        reinit=True,
        config=config,
    )

    main(config, random_state=config["random_state"])
