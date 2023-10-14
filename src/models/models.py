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
import clip

def ned_torch(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2**0.5


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

class BERT(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(BERT, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.LM = SentenceTransformer("bert-large-nli-mean-tokens")
        self.classifier = nn.Linear(model_configs["embed_dim"], model_configs["n_classes"])
        
        self.precomputed = model_configs["freeze_layers"]

        if model_configs["freeze_layers"]:
            for param in self.LM.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        if self.precomputed:
            return self.classifier(tokenized_text)
        else:
            x = self.LM(tokenized_text)
            return self.classifier(x)
        
class GPT2(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(GPT2, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.LM = SentenceTransformer('Muennighoff/SGPT-125M-weightedmean-nli-bitfit')
        self.LM.max_sequence_length = 2048
        self.LM.tokenizer.pad_token = '[PAD]'
        self.classifier = nn.Linear(model_configs["embed_dim"], num_class)
        
        self.precomputed = model_configs["freeze_layers"]

        if model_configs["freeze_layers"]:
            for param in self.LM.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        if self.precomputed:
            return self.classifier(tokenized_text)
        else:
            x = self.LM(tokenized_text)
            return self.classifier(x)
        
class RoBERTa(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(RoBERTa, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.LM = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        self.classifier = nn.Linear(model_configs["embed_dim"], num_class)
        
        self.precomputed = model_configs["freeze_layers"]

        if model_configs["freeze_layers"]:
            for param in self.LM.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        if self.precomputed:
            return self.classifier(tokenized_text)
        else:
            x = self.LM(tokenized_text)
            return self.classifier(x)
        
class MPNET(nn.Module):
    def __init__(self, vocab_size, model_configs):
        super(MPNET, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]

        self.LM = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.classifier = nn.Linear(model_configs["embed_dim"], num_class)
        
        self.precomputed = model_configs["freeze_layers"]

        if model_configs["freeze_layers"]:
            for param in self.LM.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        if self.precomputed:
            return self.classifier(tokenized_text)
        else:
            x = self.LM(tokenized_text)
            return self.classifier(x)


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
        if self.metric in ["weighted_cosine"]:
            self.dim_weights = nn.parameter.Parameter(
                nn.init.ones_(torch.empty(self.dim)),
                requires_grad=True,
            )
            
        elif self.metric in ["weighted_L2"]:
            self.dim_weights = nn.parameter.Parameter(
                nn.init.zeros_(torch.empty(self.dim)),
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
        elif self.metric == "weighted_L2":
            prototype_distances = torch.cdist(
                2*torch.sigmoid(self.dim_weights) * embedding.float(), self.protolayer.squeeze(), p=2
            ).squeeze(1) / np.sqrt(self.dim)
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
            print("saving embeddings")
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
                    pass #print(idx)
        elif (
            config["model"]["submodel"] == "roberta"
            and config["model"]["embedding"] == "sentence"
        ):
            print("saving embeddings")
            LM = SentenceTransformer(
                "sentence-transformers/all-distilroberta-v1", device=device
            )
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
                    pass #print(idx)
        elif (
            config["model"]["submodel"] == "mpnet"
            and config["model"]["embedding"] == "sentence"
        ):
            print("saving embeddings")
            LM = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", device=device
            )
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
                    pass #print(idx)
        elif config["model"]["submodel"] == 'gpt2':
            print("saving embeddings")
            LM = SentenceTransformer('Muennighoff/SGPT-125M-weightedmean-nli-bitfit', device=device)
            LM.max_sequence_length = 2048
            #LM.tokenizer.pad_token_id = 50256
            LM.tokenizer.pad_token = '[PAD]'
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
                    pass #print(idx)
                
        elif config["model"]["submodel"] == 't5':
            print("saving embeddings")
            LM = SentenceTransformer('sentence-t5-xxl', device=device)
            #LM.tokenizer.pad_token_id = 50256
            LM.tokenizer.pad_token = '[PAD]'
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
                    pass #print(idx)
                    
        elif config["model"]["submodel"] == 'clip':
            print("saving embeddings")
            LM = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
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
                    pass#print(idx)
        else:
            raise NotImplemented

        for param in LM.parameters():
            param.requires_grad = False
        if len(embedding.size()) == 1:
            embedding = embedding.unsqueeze(0).unsqueeze(0)
        mask = torch.ones(embedding.shape)  # required for attention models
        return embedding, mask, (labels - 1).to(torch.long)

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
