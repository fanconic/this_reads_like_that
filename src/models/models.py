import torch
from torch import nn
from transformers import (
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    BertModel,
    GPT2Model,
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


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
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]
        self.metric = model_configs["similaritymeasure"]
        if "bert" in model_configs["submodel"]:
            self.embedder = BertModel.from_pretrained(
                "bert-base-uncased", num_labels=num_class
            ).last_hidden_state
        elif "gpt2" in model_configs["submodel"]:
            self.embedder = GPT2Model.from_pretrained("gpt2")
        else:
            raise NotImplemented

        self.n_prototypes = model_configs["n_prototypes"]
        self.proto_size = model_configs["proto_size"]
        self.enc_size = self.embedder.config.hidden_size
        self.attn = model_configs["attention"]
        self.dilated = model_configs["dilated"]
        self.num_filters = [self.n_prototypes // len(self.dilated)] * len(self.dilated)
        self.num_filters[0] += self.n_prototypes % len(self.dilated)

        if model_configs["freeze_layers"]:
            for param in self.embedder.parameters():
                param.requires_grad = False

        # Prototype Layer:
        self.protolayer = nn.Parameter(
            nn.init.uniform_(
                torch.empty(1, self.n_prototypes, self.enc_size, self.proto_size)
            ),
            requires_grad=True,
        )

        # Classify according to similarity
        self.fc = nn.Linear(self.n_prototypes, num_class, bias=False)

    def forward(self, tokenized_text, attention_mask):
        embedding = self.embedder(
            tokenized_text, attention_mask=attention_mask
        ).last_hidden_state
        distances = self.compute_distance(embedding, attention_mask)
        prototype_distances = torch.cat(
            [torch.min(dist, dim=2)[0] for dist in distances], dim=1
        )
        class_out = self.fc(prototype_distances)
        return class_out, prototype_distances

    def compute_distance(self, embedding, mask):
        """
        # Possible Todo: Implement L2 distance
        # Note that embedding.pooler_output give sequence embedding, while last_hidden_state gives embedding for each token.
        # https://github.com/huggingface/transformers/issues/7540
        if self.metric == "cosine":
            prototype_distances = -F.cosine_similarity(
                embedding.pooler_output.unsqueeze(1), self.protolayer, dim=-1
            )
        elif self.metric == "L2":
            prototype_distances = -nes_torch(
                embedding.pooler_output.unsqueeze(1), self.protolayer, dim=-1
            )
        else:
            raise NotImplemented
        return prototype_distances"""
        N, S = embedding.shape[0:2]  # Batch size, Sequence length
        E = self.enc_size  # Encoding size
        K = self.proto_size  # Patch length
        p = self.protolayer.view(1, self.n_prototypes, 1, K * E)
        distances = []
        if self.attn:
            c = torch.combinations(torch.arange(S), r=K)
            C = c.shape[0]
            b = embedding[:, c, :].view(N, 1, C, K * E)
            if self.metric == "L2":
                dist = -nes_torch(b, p, dim=-1)
            elif self.metric == "cosine":
                dist = -F.cosine_similarity(b, p, dim=-1)
            distances.append(dist)
        else:
            j = 0
            for d, n in zip(self.dilated, self.num_filters):
                H = S - d * (K - 1)  # Number of patches
                x = embedding.unsqueeze(1)
                # use sliding window to get patches
                x = F.unfold(x, kernel_size=(K, 1), dilation=d)
                x = x.view(N, 1, H, K * E)
                p_ = p[:, j : j + n, :]
                p_ = p_.view(1, n, 1, K * E)
                if self.metric == "L2":
                    dist = -nes_torch(x, p_, dim=-1)
                elif self.metric == "cosine":
                    dist = -F.cosine_similarity(x, p_, dim=-1)
                # cut off combinations that contain padding, still keep for every example at least one combination, even
                # if it contains padding
                overlap = d * (K - 1)
                m = mask[:, overlap:].unsqueeze(1)
                m[:, :, 0] = 1
                dist = dist * m
                distances.append(dist)
                j += n

        return distances

    def get_dist(self, embedding, _):
        distances = self.compute_distance(embedding)
        return distances, []

    def get_protos(self):
        return self.protolayer.squeeze()

    def get_proto_weights(self):
        return self.fc.weight.T.cpu().detach().numpy()
