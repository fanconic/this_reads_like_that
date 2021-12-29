import torch
from torch import nn
from torch._C import device
from transformers import (
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    BertModel,
    GPT2Model,
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np


def ned_torch(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)


class MLP(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(MLP, self).__init__()
        embed_dim = model_configs["embed_dim"]
        num_class = model_configs["n_classes"]

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class GPT2(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(GPT2, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.gpt2_classifier = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", num_labels=num_class
        )
        self.gpt2_classifier.config.pad_token_id = 50256
        if model_configs["freeze_layers"]:
            for param in self.gpt2_classifier.base_model.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        return self.gpt2_classifier(tokenized_text, attention_mask=attention_mask)


class BERT(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(BERT, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.bert_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_class
        )
        if model_configs["freeze_layers"]:
            for param in self.bert_classifier.base_model.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        # print(tokenized_text.shape)
        return self.bert_classifier(tokenized_text)


class ProtoNet(nn.Module):
    # Sentence Embedding
    def __init__(self, vocab_size, model_configs):
        super(ProtoNet, self).__init__()
        num_class = model_configs["n_classes"]
        self.metric = model_configs["similaritymeasure"]
        self.n_prototypes = model_configs["n_prototypes"]
        self.dim = model_configs["embed_dim"]

        # Prototype Layer:
        self.protolayer = nn.parameter.Parameter(
            nn.init.uniform_(torch.empty(1, self.n_prototypes, self.dim), -1, 1),
            requires_grad=True,
        )
        if self.metric == "weighted_cosine":
            self.dim_weights = nn.parameter.Parameter(
                nn.init.ones_(torch.empty(self.dim)),
                requires_grad=True,
            )

        if self.metric == "learned":
            self.W = nn.parameter.Parameter(
                nn.init.uniform_(torch.empty(self.dim, self.dim)),
                requires_grad=True,
            )

        # Classify according to similarity
        self.fc = nn.Linear(self.n_prototypes, num_class, bias=False)

    def forward(self, embedding, attention_mask):
        prototype_distances = self.compute_distance(embedding.unsqueeze(1))
        class_out = self.fc(prototype_distances)
        return class_out, prototype_distances

    def compute_distance(self, embedding):
        if self.metric == "cosine":
            prototype_distances = -F.cosine_similarity(
                embedding, self.protolayer, dim=-1
            )
        elif self.metric == "L2":
            # prototype_distances = -nes_torch(embedding, self.protolayer, dim=-1)
            prototype_distances = torch.cdist(
                embedding.float(), self.protolayer.squeeze(), p=2
            ).squeeze(1) / np.sqrt(self.dim)
        elif self.metric == "L1":
            # prototype_distances = -nes_torch(embedding, self.protolayer, dim=-1)
            prototype_distances = (
                torch.cdist(embedding.float(), self.protolayer.squeeze(), p=1).squeeze(
                    1
                )
                / self.dim
            )
        elif self.metric == "weighted_cosine":
            prototype_distances = -torch.sum(
                self.dim_weights * embedding * self.protolayer, dim=-1
            ) / torch.maximum(
                (
                    torch.sqrt(
                        torch.sum(self.dim_weights * torch.square(embedding), dim=-1)
                    )
                    * torch.sqrt(
                        torch.sum(
                            self.dim_weights * torch.square(self.protolayer), dim=-1
                        )
                    )
                ),
                torch.tensor(1e-8),
            )
        elif self.metric == "dot_product":
            # exp(-x.T*y)
            prototype_distances = torch.sum(self.protolayer * embedding, dim=-1)
        elif self.metric == "learned":
            # x.T*W*y
            hW = torch.matmul(embedding, (self.W / torch.linalg.norm(self.W)))
            prototype_distances = torch.sum(hW * self.protolayer, dim=-1)
        else:
            raise NotImplemented
        return prototype_distances

    def get_dist(self, embedding, _):
        distances = self.compute_distance(embedding)
        return distances, []

    def get_protos(self):
        return self.protolayer.squeeze()

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()

    def compute_embedding(self, x, config, device):
        if (
            config["model"]["submodel"] == "bert"
            and config["model"]["embedding"] == "sentence"
        ):
            LM = SentenceTransformer("bert-large-nli-mean-tokens", device=device)
            labels = torch.empty((len(x)))
            embedding = torch.empty((len(x), config["model"]["embed_dim"]))
            for idx, (label, input) in enumerate(x):
                labels[idx] = label
                embedding[idx] = (
                    LM.encode(input, convert_to_tensor=True, device=device)
                    .cpu()
                    .detach()
                )
                if idx % 100 == 0:
                    print(idx)
        else:
            raise NotImplemented

        for param in LM.parameters():
            param.requires_grad = False
        if len(embedding.size()) == 1:
            embedding = embedding.unsqueeze(0).unsqueeze(0)
        mask = torch.ones(embedding.shape)  # required for attention models
        return embedding, mask, (labels - 1)

    @staticmethod
    def nearest_neighbors(distances, _, text_train, labels_train):
        distances = torch.cat(distances)
        _, nearest_ids = torch.topk(distances, 1, dim=0, largest=False)
        nearest_ids = nearest_ids.cpu().detach().numpy().T
        proto_id = [
            f"P{proto + 1} | sentence {index} | label {labels_train[index]} | text: "
            for proto, sent in enumerate(nearest_ids)
            for index in sent
        ]
        proto_texts = [f"{text_train[index]}" for sent in nearest_ids for index in sent]
        return proto_id, proto_texts, [nearest_ids, []]
