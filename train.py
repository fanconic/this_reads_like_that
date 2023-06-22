# Taken from this tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import os
from typing import Tuple
import torch
from torch import nn, optim
import yaml
import argparse
import random
import os
import numpy as np
import torch.nn.functional as F
import re
from src.utils.utils import (
    proto_loss,
    load_model_and_dataloader,
    get_optimizer,
    get_scheduler,
    project,
    prototype_visualization,
    get_nearest,
    mean_pooling,
)
import IPython
from tqdm import tqdm
from src.models.models import ProtoNet
from transformers import AutoTokenizer, AutoModel
import pandas as pd


def set_seed(seed):
    """Set all random seeds
    Args:
        seed (int): integer for reproducible experiments
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(config, random_state=0):
    """Main function
    Args:
        config: configuration dictionary
        random_state (default = 0): integer for reproducible experiments
    """
    use_cuda = torch.cuda.is_available()
    print("Cuda is available: ", use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(random_state)
    verbose = config["train"]["verbose"]
    gpt2_bert_lm = config["model"]["name"] in ["gpt2", "bert_baseline"]

    # load model and data
    (
        model,
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        train_loader_unshuffled,
        test_ds,
    ) = load_model_and_dataloader(config, device)

    # prepare teh optimizer
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = config["train"]["epochs"]
    if config["train"]["run_training"]:
        for epoch in range(epochs):
            # training loop
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
            )

            # Validation loop
            val_loss = val(
                model,
                val_loader,
                criterion,
                epoch,
                epochs,
                device,
                verbose,
                gpt2_bert_lm,
            )

            # project prototypes all 5 Epochs. Start projection after 50% of epochs and let last 3 epochs be only final layer training
            if (
                (epoch + 1) % 5 == 0
                and config["model"]["project"]
                and (epochs * 5 // 10) < (epoch + 1) < (epochs - 3)
            ):
                with torch.no_grad():
                    model = project(config, model, train_loader, device, False)
                    assert model.protolayer.requires_grad == True
            # final projection, train only classification layer
            if (epoch + 1) == (epochs - 3):
                with torch.no_grad():
                    model = project(config, model, train_loader, device, True)
                    model.protolayer.requires_grad = False

            scheduler.step()

        torch.save(model.state_dict(), "./saved_models/best_" + config["name"] + ".pth")

    # Test the model
    if not config["train"]["run_training"]:
        model.load_state_dict(
            torch.load("./saved_models/best_" + config["name"] + ".pth")
        )

    test(model, test_loader, criterion, device, verbose, gpt2_bert_lm)

    # Visualize the prototypes
    if config["model"]["embedding"] == "sentence":
        important_words = prototype_visualization(
            config, model, train_ds, train_loader_unshuffled, device
        )

    if config["explain"]["manual_input"] != True:
        comp_and_suff(config, model, test_ds, train_ds, test_loader, train_loader_unshuffled, device)

    # Create explanation CSV
    explain(
        config,
        model,
        test_ds,
        train_ds,
        test_loader,
        train_loader_unshuffled,
        important_words,
        device,
    )

    # Check the faithfullness. Only if we did not add manual sentences as otherwise faithfulness is distorted by them
    if config["explain"]["manual_input"] != True:
        for i in range(1, 4):
            faithful(config, model, test_ds, test_loader, device, k=i)


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
):
    """Main training loop, where the network is trained
    Args:
        model: our pytorch model
        train_loader: loader with the training data
        optimizer: optimizer for backpropagation
        criterion: loss function
        epoch:  current epoch
        epochs: max number of epochs
        device: current device (cpu or gpu)
        verbose: if the training is printed
        gpt2_bert_lm: true if we use such backbones
    """
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
        with torch.no_grad():
            model.fc.weight.copy_(model.fc.weight.clamp(max=0.0))
            if hasattr(model, "dim_weights"):
                model.dim_weights.copy_(model.dim_weights.clamp(min=0.0))
        # calculate metric
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if verbose:
            train_loader.set_description(f"Epoch [{epoch}/{epochs}]")
            train_loader.set_postfix(loss=loss.item(), acc=total_acc / total_count)

    
    print(
        "| epoch {:3d} | training accuracy {:8.3f}".format(
            epoch, total_acc / total_count
        )
    )


def val(model, val_loader, criterion, epoch, epochs, device, verbose, gpt2_bert_lm):
    """Main validation loop, where the network is validated during trianing
    Args:
        model: our pytorch model
        val_loader: loader with the validation data
        criterion: loss function
        epoch:  current epoch
        epochs: max number of epochs
        device: current device (cpu or gpu)
        verbose: if the training is printed
        gpt2_bert_lm: true if we use such backbones
    """
    val_total_acc, val_total_count, val_losses = 0, 0, []
    if verbose:
        val_loader = tqdm(val_loader)
    # Validation Loop
    model.eval()
    with torch.no_grad():
        for idx, (label, text, mask) in enumerate(val_loader):
            text, label, mask = text.to(device), label.to(device), mask.to(device)
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
    val_loss = sum(val_losses) / len(val_losses)

    # end of epoch
    print(
        "| epoch {:3d} | validation accuracy {:8.3f}".format(
            epoch, val_total_acc / val_total_count
        )
    )

    return val_loss


def test(model, test_loader, criterion, device, verbose, gpt2_bert_lm):
    """Main test loop, where the network is tested in the end
    Args:
        model: our pytorch model
        test_loader: loader with the validation data
        criterion: loss function
        device: current device (cpu or gpu)
        verbose: if the training is printed
        gpt2_bert_lm: true if we use such backbones
    """
    # Test the model
    if verbose:
        test_loader = tqdm(test_loader)
    model.eval()
    total_acc, total_count = 0, 0

    test_losses = []
    with torch.no_grad():
        df = pd.DataFrame()
        outcomes = []
        predictions = []
        for idx, (label, text, mask) in enumerate(test_loader):
            text, label, mask = text.to(device), label.to(device), mask.to(device)

            if isinstance(model, ProtoNet):
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
            predictions.append(predicted_label.argmax(1))
            outcomes.append(label)
            total_count += label.size(0)

        predictions = torch.concat(predictions)
        outcomes = torch.concat(outcomes)
        df["outcomes"] = outcomes.cpu().numpy()
        df["predictions"] = predictions.cpu().numpy()

        lower, upper = bootstrap(df)
        print("Test Loss: ", sum(test_losses) / len(test_losses))
        print(f"Test Accuracy: {total_acc / total_count:.4f} (95%-CI: {lower: .4f}, {upper:.4f})", )


def bootstrap(df) -> Tuple:
    """Bootstrap for calculating the confidence interval of a metric function
    Args:
        df (pd.DataFrame): dataframe containing 'predictions' and ' outcomes'
        func (function): metric function that takes (y_true, y_pred) as parameters
    Returns:
        lower, upper 95% confidence interval
        full bootstrap
    """
    aucs = []
    for i in range(1000):
        sample = df.sample(
            n=df.shape[0], random_state=i, replace=True
        )  # take 80% for the bootstrap
        aucs.append((sample["outcomes"] == sample["predictions"]).sum() /len(df))

    return np.percentile(np.array(aucs), 2.5), np.percentile(np.array(aucs), 97.5)


def bootstrap_faithfulness(df) -> Tuple:
    """Bootstrap for calculating the confidence interval of a metric function
    Args:
        df: dataframe containing 'predictions' and ' outcomes'
        
    Returns:
        lower, upper 95% confidence interval
        full bootstrap
    """
    aucs = []
    for i in range(1000):
        sample = pd.DataFrame(df).sample(
            n=df.shape[0], random_state=i, replace=True
        )  # take 80% for the bootstrap
        aucs.append(sample.mean())

    return np.percentile(np.array(aucs), 2.5), np.percentile(np.array(aucs), 97.5)


def explain(
    config,
    model,
    test_ds,
    train_ds,
    test_loader,
    train_batches_unshuffled,
    important_words,
    device,
):
    """Create explanation CSV file of the model, using the first 50 testing samples
    Args:
        config: configuration dictionary
        model: classification model
        test_ds: test dataset (contains the texts)
        train_ds: training dataset (contains the texts)
        test_loader: test loader, which contains the embeddings and masks
        train_batches_unshuffled: train loader, unshuffled, with embeddings and masks
        important_words: list of lists with the most important words of the prototypes
        device: current device
    """
    model.eval()
    text_train = []
    labels_train = []
    text_test = []
    labels_test = []
    embedding_test = torch.Tensor([])
    mask_test = torch.Tensor([])

    # Extract all the texts and embeddings to loop through them
    for y, x in train_ds:
        labels_train.append(y - 1)
        text_train.append(x)
    for y, x in test_ds:
        labels_test.append(y - 1)
        text_test.append(x)
    for _, x, m in test_loader:
        embedding_test = torch.cat([embedding_test, x])
        mask_test = torch.cat([mask_test, m])

    # Get the prototypes
    _, proto_texts, _ = get_nearest(
        model, train_batches_unshuffled, text_train, labels_train, device
    )
    weights = model.get_proto_weights()
    explained_test_samples = []
    with torch.no_grad():
        # Create the first values, which are a description how to read the CSV
        values = [
            f"test sample \n",
            f"true label \n",
            f"predicted label \n",
            f"probability class 0 \n",
            f"probability class 1 \n",
        ]
        for j in range(config["explain"]["max_numbers"]):
            values.append(f"explanation_{j + 1} \n")
            values.append(f"keywords_prototype_{j+1} \n")
            values.append(f"keywords_sentence_{j+1} \n")
            values.append(f"score_{j + 1} \n")
            values.append(f"id_{j + 1} \n")
            values.append(f"similarity_{j + 1} \n")
            values.append(f"weight_{j + 1} \n")
        explained_test_samples.append(values)
        # Initialize for Test & Protovisualization
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
        elif (
            config["model"]["embedding"] == "sentence"
            and config["model"]["submodel"] == "roberta"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-distilroberta-v1"
            )
            model_emb = AutoModel.from_pretrained(
                "sentence-transformers/all-distilroberta-v1"
            )
        elif (
            config["model"]["embedding"] == "sentence"
            and config["model"]["submodel"] == "mpnet"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )
            model_emb = AutoModel.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )

        if config["explain"]["manual_input"] == True:
            manual_sentences = [
                "NLP is a really interesting course!",
                "NLP is not an interesting course!",
                "NLP is not not an interesting course!",
                "The movie's plot was just mediocre, but the stunning performance by Joaquin Phoenix will make the flick a hit.",
            ]
            tokenized_proto = tokenizer(
                manual_sentences, padding=True, truncation=True, return_tensors="pt"
            )

            # Compute token embeddings
            with torch.no_grad():
                model_output = model_emb(**tokenized_proto)
            # Perform pooling. In this case, mean pooling.
            manual_sentence_embeddings = mean_pooling(
                model_output, tokenized_proto["attention_mask"]
            ).to(device)
            for z in range(len(manual_sentences)):
                mask = (
                    tokenized_proto["attention_mask"][z]
                    .to(device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                emb_manual = (
                    manual_sentence_embeddings[z].to(device).unsqueeze(0).unsqueeze(0)
                )
                predicted_label, prototype_distances = model.forward(emb_manual, mask)
                predicted = (
                    torch.argmax(predicted_label, dim=-1).squeeze().cpu().detach()
                )
                probability = (
                    torch.nn.functional.softmax(predicted_label, dim=-1)
                    .squeeze()
                    .tolist()
                )
                similarity_score = (
                    prototype_distances.cpu().detach().squeeze()
                    * weights[:, predicted].T
                )
                top_scores = similarity_score
                # Sort the best scoring prototypes and add them to the explanation CSV
                sorted = torch.argsort(top_scores, descending=True)
                # Create Variations of all Sentence Embeddings by removing one word
                keep_words = []
                nearest_vals, _ = model.get_dist(emb_manual, _)
                for nth_proto in range(config["explain"]["max_numbers"]):
                    text_strings = manual_sentences[z]
                    nearest_val_proto = (
                        nearest_vals[:, sorted[nth_proto]].cpu().detach().numpy()
                    )

                    top_words = np.min(
                        (5, len(re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)))
                    )
                    text_words = []
                    text_distance = np.empty(top_words)

                    for nth_removed_word in range(
                        top_words
                    ):  # Iteratively remove most important words
                        text = re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)
                        sentence_variants = [
                            text[:i] + text[i + 1 :] for i in range(len(text))
                        ]
                        left_word = [text[i] for i in range(len(text))]
                        sentence_variants = [" ".join(i) for i in sentence_variants]
                        tokenized_text = tokenizer(
                            sentence_variants,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        # Compute token embeddings
                        with torch.no_grad():
                            model_output = model_emb(**tokenized_text)
                        # Perform pooling. In this case, mean pooling.
                        sentence_embeddings = mean_pooling(
                            model_output, tokenized_text["attention_mask"]
                        ).to(device)
                        # Calculate distance to orginial embedding of sentence.
                        dist_per_word, _ = model.get_dist(
                            sentence_embeddings.unsqueeze(1), _
                        )
                        dist_per_word = dist_per_word[:, sorted[nth_proto]]
                        farthest_val, farthest_ids = torch.topk(
                            dist_per_word, 1, dim=0, largest=True
                        )  # Store largest distance
                        text_words.append(left_word[farthest_ids])
                        text_distance[nth_removed_word] = farthest_val
                        text_strings = sentence_variants[farthest_ids]

                    # Choose words that give 75% of distance of all 5 words
                    proto_word_dist = text_distance - nearest_val_proto
                    # Check that removing words made words move away
                    if proto_word_dist[-1] < 0:
                        cutoff = proto_word_dist >= 0
                    else:
                        cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
                    # Include the word responsible for the 75% drop
                    cutoff[sum(cutoff)] = True
                    keep_words.append([text_words[i] for i in np.where(cutoff)[0]])

                # Calculate important words of prototypes wrt Sentence
                protos_words = []
                for nth_proto in range(config["explain"]["max_numbers"]):
                    proto_strings = proto_texts[sorted[nth_proto]]
                    nearest_val_proto = (
                        nearest_vals[:, sorted[nth_proto]].cpu().detach().numpy()
                    )
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
                            prototype[:i] + prototype[i + 1 :]
                            for i in range(len(prototype))
                        ]
                        left_word = [prototype[i] for i in range(len(prototype))]
                        sentence_variants = [" ".join(i) for i in sentence_variants]
                        tokenized_proto = tokenizer(
                            sentence_variants,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        # Compute token embeddings
                        with torch.no_grad():
                            model_output = model_emb(**tokenized_proto)
                        # Perform pooling. In this case, mean pooling.
                        sentence_embeddings = mean_pooling(
                            model_output, tokenized_proto["attention_mask"]
                        ).to(device)
                        # Calculate distance to of truncated prototype to sentence.

                        if config["model"]["similaritymeasure"] == "weighted_cosine":
                            dist_per_word = -torch.sum(
                                model.dim_weights * sentence_embeddings * emb_manual,
                                dim=-1,
                            ) / torch.maximum(
                                (
                                    torch.sqrt(
                                        torch.sum(
                                            model.dim_weights
                                            * torch.square(sentence_embeddings),
                                            dim=-1,
                                        )
                                    )
                                    * torch.sqrt(
                                        torch.sum(
                                            model.dim_weights
                                            * torch.square(emb_manual),
                                            dim=-1,
                                        )
                                    )
                                ),
                                torch.tensor(1e-8),
                            )
                        elif config["model"]["similaritymeasure"] == "cosine":
                            dist_per_word = -F.cosine_similarity(
                                emb_manual, sentence_embeddings, dim=-1
                            )
                        elif config["model"]["similaritymeasure"] == "L2":
                            # prototype_distances = -nes_torch(embedding, self.protolayer, dim=-1)
                            dist_per_word = (
                                torch.cdist(
                                    sentence_embeddings.float(), emb_manual, p=2
                                ).squeeze(1)
                                / np.sqrt(model.dim)
                            ).squeeze(-1)
                        elif config["model"]["similaritymeasure"] == "L1":
                            dist_per_word = (
                                torch.cdist(
                                    sentence_embeddings.float(), emb_manual, p=1
                                ).squeeze(1)
                                / model.dim
                            ).squeeze(-1)
                        elif config["model"]["similaritymeasure"] == "dot_product":
                            # exp(-x.T*y)
                            dist_per_word = torch.sum(
                                emb_manual * sentence_embeddings, dim=-1
                            )
                        elif config["model"]["similaritymeasure"] == "learned":
                            # x.T*W*y
                            hW = torch.matmul(
                                emb_manual, (model.W / torch.linalg.norm(model.W))
                            )
                            dist_per_word = torch.sum(hW * sentence_embeddings, dim=-1)
                        else:
                            raise NotImplemented

                        farthest_val, farthest_ids = torch.topk(
                            dist_per_word.squeeze(0), 1, dim=0, largest=True
                        )  # Store largest distance
                        proto_words.append(left_word[farthest_ids])
                        proto_distance[nth_removed_word] = farthest_val
                        proto_strings = sentence_variants[farthest_ids]

                    # Choose words that give 75% of distance of all 5 words
                    proto_word_dist = proto_distance - nearest_val_proto
                    # Check that removing words made words move away
                    if proto_word_dist[-1] < 0:
                        cutoff = proto_word_dist >= 0
                    else:
                        cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
                    # Include the word responsible for the 75% drop
                    cutoff[sum(cutoff)] = True
                    protos_words.append([proto_words[i] for i in np.where(cutoff)[0]])

                values = [
                    "".join(manual_sentences[z]) + "\n",
                    f"{int(10)}\n",
                    f"{int(predicted)}\n",
                    f"{probability[0]:.3f}\n",
                    f"{probability[1]:.3f}\n",
                ]
                for i, j in enumerate(sorted):
                    idx = j.item()
                    nearest_proto = proto_texts[idx]
                    values.append(f"{nearest_proto}\n")
                    values.append(", ".join(protos_words[i]) + "\n")
                    values.append(", ".join(keep_words[i]) + "\n")
                    values.append(f"{float(top_scores[j]):.3f}\n")
                    values.append(f"{idx + 1}\n")
                    values.append(f"{float(-prototype_distances.squeeze()[idx]):.3f}\n")
                    values.append(f"{float(-weights[idx, predicted]):.3f}\n")
                    if i == config["explain"]["max_numbers"] - 1:
                        break
                explained_test_samples.append(values)

        # Go through all the test samples and show their closest prototypes
        for i in range(len(labels_test)):

            emb = embedding_test[i].to(device).unsqueeze(0).unsqueeze(0)
            mask = mask_test[i].to(device).unsqueeze(0).unsqueeze(0)
            predicted_label, prototype_distances = model.forward(emb, mask)
            predicted = torch.argmax(predicted_label, dim=-1).squeeze().cpu().detach()
            probability = (
                torch.nn.functional.softmax(predicted_label, dim=-1).squeeze().tolist()
            )
            similarity_score = (
                prototype_distances.cpu().detach().squeeze() * weights[:, predicted].T
            )
            top_scores = similarity_score

            # Sort the best scoring prototypes and add them to the explanation CSV
            sorted = torch.argsort(top_scores, descending=True)
            if i <= 50:
                print(i)
                from datetime import datetime

                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
                # Create Variations of all Sentence Embeddings by removing one word
                keep_words = []
                nearest_vals, _ = model.get_dist(emb, _)
                for nth_proto in range(config["explain"]["max_numbers"]):
                    text_strings = text_test[i]
                    nearest_val_proto = (
                        nearest_vals[:, sorted[nth_proto]].cpu().detach().numpy()
                    )

                    top_words = np.min(
                        (5, len(re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)))
                    )
                    text_words = []
                    text_distance = np.empty(top_words)

                    for nth_removed_word in range(
                        top_words
                    ):  # Iteratively remove most important words
                        text = re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)
                        sentence_variants = [
                            text[:i] + text[i + 1 :] for i in range(len(text))
                        ]
                        left_word = [text[i] for i in range(len(text))]
                        sentence_variants = [" ".join(i) for i in sentence_variants]
                        tokenized_text = tokenizer(
                            sentence_variants,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        # Compute token embeddings
                        with torch.no_grad():
                            model_output = model_emb(**tokenized_text)
                        # Perform pooling. In this case, mean pooling.
                        sentence_embeddings = mean_pooling(
                            model_output, tokenized_text["attention_mask"]
                        ).to(device)
                        # Calculate distance to orginial embedding of sentence.
                        dist_per_word, _ = model.get_dist(
                            sentence_embeddings.unsqueeze(1), _
                        )
                        dist_per_word = dist_per_word[:, sorted[nth_proto]]
                        farthest_val, farthest_ids = torch.topk(
                            dist_per_word, 1, dim=0, largest=True
                        )  # Store largest distance
                        text_words.append(left_word[farthest_ids])
                        text_distance[nth_removed_word] = farthest_val
                        text_strings = sentence_variants[farthest_ids]

                    # Choose words that give 75% of distance of all 5 words
                    proto_word_dist = text_distance - nearest_val_proto
                    # Check that removing words made words move away
                    if proto_word_dist[-1] < 0:
                        cutoff = proto_word_dist >= 0
                    else:
                        cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
                    # Include the word responsible for the 75% drop
                    cutoff[sum(cutoff)] = True
                    keep_words.append([text_words[i] for i in np.where(cutoff)[0]])

                # Calculate important words of prototypes wrt Sentence
                protos_words = []
                for nth_proto in range(config["explain"]["max_numbers"]):
                    proto_strings = proto_texts[sorted[nth_proto]]
                    nearest_val_proto = (
                        nearest_vals[:, sorted[nth_proto]].cpu().detach().numpy()
                    )
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
                            prototype[:i] + prototype[i + 1 :]
                            for i in range(len(prototype))
                        ]
                        left_word = [prototype[i] for i in range(len(prototype))]
                        sentence_variants = [" ".join(i) for i in sentence_variants]
                        tokenized_proto = tokenizer(
                            sentence_variants,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        # Compute token embeddings
                        with torch.no_grad():
                            model_output = model_emb(**tokenized_proto)
                        # Perform pooling. In this case, mean pooling.
                        sentence_embeddings = mean_pooling(
                            model_output, tokenized_proto["attention_mask"]
                        ).to(device)
                        # Calculate distance to of truncated prototype to sentence.

                        if config["model"]["similaritymeasure"] == "weighted_cosine":
                            dist_per_word = -torch.sum(
                                model.dim_weights * sentence_embeddings * emb, dim=-1
                            ) / torch.maximum(
                                (
                                    torch.sqrt(
                                        torch.sum(
                                            model.dim_weights
                                            * torch.square(sentence_embeddings),
                                            dim=-1,
                                        )
                                    )
                                    * torch.sqrt(
                                        torch.sum(
                                            model.dim_weights * torch.square(emb),
                                            dim=-1,
                                        )
                                    )
                                ),
                                torch.tensor(1e-8),
                            )
                        elif config["model"]["similaritymeasure"] == "cosine":
                            dist_per_word = -F.cosine_similarity(
                                emb, sentence_embeddings, dim=-1
                            )
                        elif config["model"]["similaritymeasure"] == "L2":
                            # prototype_distances = -nes_torch(embedding, self.protolayer, dim=-1)
                            dist_per_word = (
                                torch.cdist(
                                    sentence_embeddings.float(), emb, p=2
                                ).squeeze(1)
                                / np.sqrt(model.dim)
                            ).squeeze(-1)
                        elif config["model"]["similaritymeasure"] == "L1":
                            dist_per_word = (
                                torch.cdist(
                                    sentence_embeddings.float(), emb, p=1
                                ).squeeze(1)
                                / model.dim
                            ).squeeze(-1)
                        elif config["model"]["similaritymeasure"] == "dot_product":
                            # exp(-x.T*y)
                            dist_per_word = torch.sum(emb * sentence_embeddings, dim=-1)
                        elif config["model"]["similaritymeasure"] == "learned":
                            # x.T*W*y
                            hW = torch.matmul(
                                emb, (model.W / torch.linalg.norm(model.W))
                            )
                            dist_per_word = torch.sum(hW * sentence_embeddings, dim=-1)
                        else:
                            raise NotImplemented

                        farthest_val, farthest_ids = torch.topk(
                            dist_per_word.squeeze(0), 1, dim=0, largest=True
                        )  # Store largest distance
                        proto_words.append(left_word[farthest_ids])
                        proto_distance[nth_removed_word] = farthest_val
                        proto_strings = sentence_variants[farthest_ids]

                    # Choose words that give 75% of distance of all 5 words
                    proto_word_dist = proto_distance - nearest_val_proto
                    # Check that removing words made words move away
                    if proto_word_dist[-1] < 0:
                        cutoff = proto_word_dist >= 0
                    else:
                        cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
                    # Include the word responsible for the 75% drop
                    cutoff[sum(cutoff)] = True
                    protos_words.append([proto_words[i] for i in np.where(cutoff)[0]])

                values = [
                    "".join(text_test[i]) + "\n",
                    f"{int(labels_test[i])}\n",
                    f"{int(predicted)}\n",
                    f"{probability[0]:.3f}\n",
                    f"{probability[1]:.3f}\n",
                ]
                for i, j in enumerate(sorted):
                    idx = j.item()
                    nearest_proto = proto_texts[idx]
                    values.append(f"{nearest_proto}\n")
                    values.append(", ".join(protos_words[i]) + "\n")
                    values.append(", ".join(keep_words[i]) + "\n")
                    values.append(f"{float(top_scores[j]):.3f}\n")
                    values.append(f"{idx + 1}\n")
                    values.append(f"{float(-prototype_distances.squeeze()[idx]):.3f}\n")
                    values.append(f"{float(-weights[idx, predicted]):.3f}\n")
                    if i == config["explain"]["max_numbers"] - 1:
                        break

            else:

                values = [
                    "".join(text_test[i]) + "\n",
                    f"{int(labels_test[i])}\n",
                    f"{int(predicted)}\n",
                    f"{probability[0]:.3f}\n",
                    f"{probability[1]:.3f}\n",
                ]
                for i, j in enumerate(sorted):
                    idx = j.item()
                    nearest_proto = proto_texts[idx]
                    values.append(f"{nearest_proto}\n")
                    values.append(", ".join(important_words[idx]) + "\n")
                    values.append(", ".join(" ") + "\n")
                    values.append(f"{float(top_scores[j]):.3f}\n")
                    values.append(f"{idx + 1}\n")
                    values.append(f"{float(-prototype_distances.squeeze()[idx]):.3f}\n")
                    values.append(f"{float(-weights[idx, predicted]):.3f}\n")
                    if i == config["explain"]["max_numbers"] - 1:
                        break

            explained_test_samples.append(values)

    import csv

    # Write everything to CSV
    save_path = os.path.join("./explanations", config["name"] + "_explained.csv")
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(explained_test_samples)


def faithful(config, model, test_ds, test_loader, device, k=1):
    """Computes the faithfullness of the Model, once the highest prototype is removed
    Args:
        config: configuration dictionary
        model: classification model
        test_ds: test dataset (contains the texts)
        test_loader: test loader, which contains the embeddings and masks
        device: current device
        k (default 1): number of prototypes to be removed
    """
    # Extract all the texts and embeddings to loop through them
    text_test = []
    labels_test = []
    embedding_test = torch.Tensor([])
    mask_test = torch.Tensor([])
    for y, x in test_ds:
        labels_test.append(y - 1)
        text_test.append(x)
    for _, x, m in test_loader:
        embedding_test = torch.cat([embedding_test, x])
        mask_test = torch.cat([mask_test, m])

    # Read the csv that was previously created in explain()
    data_explained = pd.read_csv(
        os.path.join("./explanations", config["name"] + "_explained.csv")
    )

    score = [f"id_{i} \n" for i in range(1, config["explain"]["max_numbers"] + 1)]

    # extract the top k prototypes
    top_ids = torch.tensor(data_explained[score].to_numpy())[:, :k] - 1

    # Create a new
    explained_test_samples = []
    values = []
    all_preds = []
    with torch.no_grad():
        # Create the explanation of the CSV first
        values.append(f"test sample \n")
        values.append(f"true label \n")
        values.append(f"predicted label \n")
        values.append(f"probability class 0 \n")
        values.append(f"probability class 1 \n")
        explained_test_samples.append(values)

        # Iterate through all the test samples
        for i in range(len(labels_test)):
            weights = model.fc.weight.detach().clone()
            tmp = weights.clone()

            # set the weights of the best prototype to zero and make a prediction
            weights[:, top_ids[i]] = torch.zeros(config["model"]["n_classes"], 1).to(
                device
            )
            model.fc.weight.copy_(weights)
            emb = embedding_test[i].to(device).unsqueeze(0)
            mask = mask_test[i].to(device).unsqueeze(0)
            predicted_label, prototype_distances = model.forward(emb, mask)
            model.fc.weight.copy_(tmp)
            predicted = torch.argmax(predicted_label).cpu().detach()
            probability = (
                torch.nn.functional.softmax(predicted_label, dim=1).squeeze().tolist()
            )
            all_preds += [predicted.cpu().detach().numpy().tolist()]

            # Append values for the CSV
            values = [
                "".join(text_test[i]) + "\n",
                f"{int(labels_test[i])}\n",
                f"{int(predicted)}\n",
                f"{probability[0]:.3f}\n",
                f"{probability[1]:.3f}\n",
            ]
            explained_test_samples.append(values)

    # Calculate new Accuracy
    acc_test = (np.array(labels_test) == np.array(all_preds)).sum() / len(labels_test)
    print(f"New Accuracy without {k} best prototypes: {acc_test * 100}")

def comp_and_suff(
    config,
    model,
    test_ds,
    train_ds,
    test_loader,
    train_batches_unshuffled,
    device,
):
    """Calculate comprehensiveness and sufficiency according to ERASER
    Args:
        config: configuration dictionary
        model: classification model
        test_ds: test dataset (contains the texts)
        train_ds: training dataset (contains the texts)
        test_loader: test loader, which contains the embeddings and masks
        train_batches_unshuffled: train loader, unshuffled, with embeddings and masks
        device: current device
    """
    model.eval()
    text_train = []
    labels_train = []
    text_test = []
    labels_test = []
    embedding_test = torch.Tensor([])
    mask_test = torch.Tensor([])

    # Extract all the texts and embeddings to loop through them
    for y, x in train_ds:
        labels_train.append(y - 1)
        text_train.append(x)
    for y, x in test_ds:
        labels_test.append(y - 1)
        text_test.append(x)
    for _, x, m in test_loader:
        embedding_test = torch.cat([embedding_test, x])
        mask_test = torch.cat([mask_test, m])

    weights = model.get_proto_weights()

    with torch.no_grad():
        # Initialize for Test & Protovisualization
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
        elif (
            config["model"]["embedding"] == "sentence"
            and config["model"]["submodel"] == "roberta"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-distilroberta-v1"
            )
            model_emb = AutoModel.from_pretrained(
                "sentence-transformers/all-distilroberta-v1"
            )
        elif (
            config["model"]["embedding"] == "sentence"
            and config["model"]["submodel"] == "mpnet"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )
            model_emb = AutoModel.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )
        
        # Go through all the test samples and remove rationals of closest prototype
        from datetime import datetime
        keep_words = []
        removed_sentences = []
        for i in range(len(labels_test)):
            if i%50==0:
                print(i)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
            emb = embedding_test[i].to(device).unsqueeze(0).unsqueeze(0)
            mask = mask_test[i].to(device).unsqueeze(0).unsqueeze(0)
            predicted_label, prototype_distances = model.forward(emb, mask)
            predicted = torch.argmax(predicted_label, dim=-1).squeeze().cpu().detach()
            probability = (
                torch.nn.functional.softmax(predicted_label, dim=-1).squeeze().tolist()
            )
            similarity_score = (
                prototype_distances.cpu().detach().squeeze() * weights[:, predicted].T
            )
            top_scores = similarity_score

            # Sort the best scoring prototypes with respect to prediction (not ground truth!)
            sorted_protos = torch.argsort(top_scores, descending=True)
            
            # Store avg distance to 3 closest prototypes
            nearest_vals, _ = model.get_dist(emb, _)
            text_strings = text_test[i]
            nearest_val_proto = (
                nearest_vals[:, sorted_protos[0:3]].cpu().detach().numpy()
            )

            # Create Variations of all Sentence Embeddings by removing one word
            top_words = np.min(
                (10, len(re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)))
            )
            text_words = []
            text_distance = np.empty(top_words)

            for nth_removed_word in range(
                top_words
            ):  # Iteratively remove most important words
                text = re.findall(r"[\w']+|[.,\":\[\]!?;]", text_strings)
                sentence_variants = [
                    text[:i] + text[i + 1 :] for i in range(len(text))
                ]
                left_word = [text[i] for i in range(len(text))]
                sentence_variants = [" ".join(i) for i in sentence_variants]
                tokenized_text = tokenizer(
                    sentence_variants,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                # Compute token embeddings
                with torch.no_grad():
                    model_output = model_emb(**tokenized_text)
                # Perform pooling. In this case, mean pooling.
                sentence_embeddings = mean_pooling(
                    model_output, tokenized_text["attention_mask"]
                ).to(device)
                # Calculate distance to orginial embedding of sentence.
                dist_per_word, _ = model.get_dist(
                    sentence_embeddings.unsqueeze(1), _
                )
                # Look at average distance over 3 closest prototypes
                dist_per_word = dist_per_word[:, sorted_protos[0:3]].mean(1)
                farthest_val, farthest_ids = torch.topk(
                    dist_per_word, 1, dim=0, largest=True
                )  # Store largest distance
                text_words.append(left_word[farthest_ids])
                text_distance[nth_removed_word] = farthest_val
                text_strings = sentence_variants[farthest_ids]
                if i > len(removed_sentences)-1:
                    removed_sentences.append([text_strings])
                else:
                    removed_sentences[i].append(text_strings)
            # Choose words that give 75% of distance of all 5 words
            proto_word_dist = text_distance - nearest_val_proto.mean()
            # Check that removing words made words move away
            if proto_word_dist[-1] < 0:
                cutoff = np.logical_and(proto_word_dist >= 0 , proto_word_dist <= 0.75 * proto_word_dist.max())
                # Include the word responsible for the 75% drop
                above_words = proto_word_dist >= 0.75 * proto_word_dist.max()
                if proto_word_dist.max() >= 0:
                    cutoff[above_words][np.argmin(proto_word_dist[above_words])] = True
            else:
                cutoff = proto_word_dist <= 0.75 * proto_word_dist[-1]
                # Include the word responsible for the 75% drop
                cutoff[sum(cutoff)] = True
            text_words = [text_words[i] for i in np.where(cutoff)[0]]
            
            # Store input, rationals, no rationals
            keep_words.append([text_test[i]])
            ordered_list = sorted(text_words, key=lambda x: text_test[i].index(x))
            delimiter = ' '
            keep_words[i].append(delimiter.join(ordered_list))
            keep_words[i].append(removed_sentences[i][sum(cutoff)-1])
            keep_words[i].append(predicted)
            keep_words[i].append(probability[predicted])

        # Computing predictions for input sentence, only rationals, sentence w/o rationals
        predicted_label_full = [sublist[3] for sublist in keep_words]
        predicted_prob_full = [sublist[4] for sublist in keep_words]
        probs = np.empty((len(predicted_label_full),3))
        for sentence_version in range(3):
            sentences = [sublist[sentence_version] for sublist in keep_words]

            tokenized_text = tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # Compute token embeddings
            with torch.no_grad():
                model_output = model_emb(**tokenized_text)
            # Perform pooling. In this case, mean pooling.
            sentence_embeddings = mean_pooling(
                model_output, tokenized_text["attention_mask"]
            ).to(device)
            predicted_label, prototype_distances = model.forward(sentence_embeddings, mask.repeat(len(sentence_embeddings),1,1))
            probability = (
                torch.nn.functional.softmax(predicted_label, dim=1).squeeze().tolist()
            )
            if sentence_version == 0:
                # assert that predicted probs are similar to the one before
                assert (np.logical_or(np.array(predicted_prob_full)-0.001 <= np.array([max(sublist) for sublist in probability]), np.array([max(sublist) for sublist in probability]) <= np.array(predicted_prob_full)+0.001)).all()
            
            prob_predicted = np.array([probability[i][predicted_label_full[i]] for i in range(len(probability))])
            probs[:,sentence_version] = prob_predicted

        comp_samplewise = (probs[:,0] - probs[:,2])
        suff_samplewise = (probs[:,0] - probs[:,1])
        
        lower_comp, upper_comp = bootstrap_faithfulness(comp_samplewise)
        lower_suff, upper_suff = bootstrap_faithfulness(suff_samplewise)
        
        comp = comp_samplewise.mean()
        suff = suff_samplewise.mean()
        print(f"Test Comprehensiveness: {comp:.4f} (95%-CI: {lower_comp: .4f}, {upper_comp:.4f})", )
        print(f"Test Sufficiency: {suff:.4f} (95%-CI: {lower_suff: .4f}, {upper_suff:.4f})", )
        print('Comprehensiveness:', comp)
        print('Sufficiency:', suff)




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

    main(config, random_state=config["random_state"])
