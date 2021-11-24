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

    # build the tokenized vocabulary:
    train_loader, vocab = build_loader(train_iter)

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
    log_interval = 500
    start_time = time.time()

    for epoch in range(config["train"]["epochs"]):
        for idx, (label, text, offsets) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(train_loader),
                                                  total_acc/total_count))
                wandb.log({"epoch": epoch,
                           "loss": loss,
                           "accuracy": total_acc/total_count})

                total_acc, total_count = 0, 0
                start_time = time.time()

    # Evaluate the model
    model.eval()
    total_acc, total_count = 0, 0

    test_losses = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(train_loader):
            predicted_label = model(text, offsets)
            test_loss = criterion(predicted_label, label)
            test_losses.append(test_loss)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

        wandb.log({"test_loss": np.mean(test_losses),
                   # "test_accuracy": total_acc/total_count
                   })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    print("Running experiment: {}".format(config["name"]))

    # Weights & Biases for tracking training
    wandb.init(
        project="nlp_groupproject",
        entity="nlp_groupproject",
        name=config["name"],
        reinit=True,
        config=config,
    )

    main(config, random_state=config["random_state"])
