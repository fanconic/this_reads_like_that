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
from src.utils.utils import (
    proto_loss,
    load_model_and_dataloader,
    get_optimizer,
    get_scheduler,
)
import time
import IPython
from tqdm import tqdm
from src.models.models import ProtoNet


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
    verbose = config["train"]["verbose"]
    gpt2_bert_lm = config["model"]["name"] in ["gpt2", "bert_baseline"]

    # load model and data
    model, train_loader, val_loader, test_loader = load_model_and_dataloader(
        wandb, config, device
    )

    # prepare teh optimizer
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = config["train"]["epochs"]
    for epoch in range(epochs):
        train(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            epochs,
            device,
            verbose,
            gpt2_bert_lm,
            scheduler,
        )
        val(model, val_loader, criterion, epoch, epochs, device, verbose, gpt2_bert_lm)
    test(model, test_loader, criterion, device, verbose, gpt2_bert_lm)


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    epoch,
    epochs,
    device,
    verbose,
    gpt2_bert_lm,
    scheduler,
):
    if verbose:
        train_loader = tqdm(train_loader)
    total_acc, total_count = 0, 0
    # Training Loop
    model.train()
    for idx, (label, text, mask) in enumerate(train_loader):
        text, label, mask = text.to(device), label.to(device), mask.to(device)
        optimizer.zero_grad()
        if isinstance(model, ProtoNet):
            predicted_label, prototype_distances = model(text, mask)
        else:
            predicted_label = model(text, mask)
        predicted_label = predicted_label.logits if gpt2_bert_lm else predicted_label
        ce_loss = criterion(predicted_label, label)
        if isinstance(model, ProtoNet):
            distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = proto_loss(
                prototype_distances, label, model, config, device
            )
            loss = (
                ce_loss
                + config["loss"]["lambda1"] * distr_loss
                + config["loss"]["lambda2"] * clust_loss
                + config["loss"]["lambda3"] * sep_loss
                + config["loss"]["lambda4"] * divers_loss
                + config["loss"]["lambda5"] * l1_loss
            )
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # calculate metric
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if verbose:
            train_loader.set_description(f"Epoch [{epoch}/{epochs}]")
            train_loader.set_postfix(loss=loss.item(), acc=total_acc / total_count)
        wandb.log({"train_loss": loss, "train_accuracy": total_acc / total_count})
    print(
        "| epoch {:3d} | training accuracy {:8.3f}".format(
            epoch, total_acc / total_count
        )
    )


def val(model, val_loader, criterion, epoch, epochs, device, verbose, gpt2_bert_lm):
    val_total_acc, val_total_count, val_losses = 0, 0, []
    if verbose:
        val_loader = tqdm(val_loader)
    # Validation Loop
    model.eval()
    with torch.no_grad():
        for idx, (label, text, mask) in enumerate(val_loader):
            text, label, mask = text.to(device), label.to(device), mask.to(device)
            optimizer.zero_grad()
            if isinstance(model, ProtoNet):
                predicted_label, prototype_distances = model(text, mask)
            else:
                predicted_label = model(text, mask)

            predicted_label = (
                predicted_label.logits if gpt2_bert_lm else predicted_label
            )
            # calc loss
            val_ce_loss = criterion(predicted_label, label)

            if isinstance(model, ProtoNet):
                distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = proto_loss(
                    prototype_distances, label, model, config, device
                )

                val_loss = (
                    val_ce_loss
                    + config["loss"]["lambda1"] * distr_loss
                    + config["loss"]["lambda2"] * clust_loss
                    + config["loss"]["lambda3"] * sep_loss
                    + config["loss"]["lambda4"] * divers_loss
                    + config["loss"]["lambda5"] * l1_loss
                )
            else:
                val_loss = val_ce_loss

            val_losses.append(val_loss)
            val_total_acc += (predicted_label.argmax(1) == label).sum().item()
            val_total_count += label.size(0)
            if verbose:
                val_loader.set_description(f"Epoch [{epoch}/{epochs}]")
                val_loader.set_postfix(
                    loss=val_loss.item(), acc=val_total_acc / val_total_count
                )
    wandb.log(
        {
            "epoch": epoch,
            "val_loss": sum(val_losses) / len(val_losses),
            "val_accuracy": val_total_acc / val_total_count,
        }
    )

    # end of epoch
    print(
        "| epoch {:3d} | validation accuracy {:8.3f}".format(
            epoch, val_total_acc / val_total_count
        )
    )


def test(model, test_loader, criterion, device, verbose, gpt2_bert_lm):
    # Test the model
    if verbose:
        test_loader = tqdm(test_loader)
    model.eval()
    total_acc, total_count = 0, 0

    test_losses = []
    with torch.no_grad():
        for idx, (label, text, mask) in enumerate(test_loader):
            text, label, mask = text.to(device), label.to(device), mask.to(device)

            if isinstance(model, Proto_BERT):
                predicted_label, prototype_distances = model(text, mask)
            else:
                predicted_label = model(text, mask)
            predicted_label = (
                predicted_label.logits if gpt2_bert_lm else predicted_label
            )
            test_ce_loss = criterion(predicted_label, label)

            if isinstance(model, ProtoNet):
                distr_loss, clust_loss, sep_loss, divers_loss, l1_loss = proto_loss(
                    prototype_distances, label, model, config, device
                )
                test_loss = (
                    test_ce_loss
                    + config["loss"]["lambda1"] * distr_loss
                    + config["loss"]["lambda2"] * clust_loss
                    + config["loss"]["lambda3"] * sep_loss
                    + config["loss"]["lambda4"] * divers_loss
                    + config["loss"]["lambda5"] * l1_loss
                )
            else:
                test_loss = test_ce_loss

            test_losses.append(test_loss)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        print("Test Loss: ", sum(test_losses) / len(test_losses))
        print("Test Accuracy: ", total_acc / total_count)
        wandb.log(
            {
                "test_loss": sum(test_losses) / len(test_losses),
                "test_accuracy": total_acc / total_count,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    config["model"]["prototype class"] = torch.eye(config["model"]["n_classes"]).repeat(
        config["model"]["n_prototypes"] // config["model"]["n_classes"], 1
    )
    print("Running experiment: {}".format(config["name"]))

    # Weights & Biases for tracking training
    mode = "online" if config["wandb_logging"] else "disabled"

    wandb.init(
        mode=mode,
        project="nlp_groupproject",
        entity="nlp_groupproject",
        name=config["name"],
        reinit=True,
        config=config,
    )

    main(config, random_state=config["random_state"])
