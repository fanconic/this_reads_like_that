import torch
from torch import nn
from transformers import (
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    BertModel,
)
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


class Proto_BERT(nn.Module):
    # Sentence Embedding

    def __init__(self, vocab_size, model_configs):
        super(Proto_BERT, self).__init__()
        # fine_tune = model_configs["fine_tune"]
        num_class = model_configs["n_classes"]
        self.metric = model_configs["similaritymeasure"]
        self.bert_embedding = BertModel.from_pretrained(
            "bert-base-uncased", num_labels=num_class
        )


        if model_configs["freeze_layers"]:
            for param in self.bert_embedding.base_model.parameters():
                param.requires_grad = False

        # Prototype Layer:
        n_prototypes = model_configs["n_prototypes"]
        self.protolayer = nn.Parameter(
            nn.init.uniform_(torch.empty(1, n_prototypes, model_configs["embed_dim"])),
            requires_grad=True,
        )

        # Classify according to similarity
        #self.test = nn.Linear(model_configs["embed_dim"],n_prototypes)
        self.fc = nn.Linear(n_prototypes, num_class, bias=False)

    def forward(self, tokenized_text, attention_mask):
        # print(tokenized_text.shape)
        embedding = self.bert_embedding(tokenized_text).pooler_output.unsqueeze(1)
        #prototype_distances = self.test(embedding.pooler_output)
        prototype_distances = self.compute_distance(embedding)
        class_out = self.fc(prototype_distances)
        return class_out, prototype_distances

    def compute_distance(self, embedding):
        # Note that embedding.pooler_output give sequence embedding, while last_hidden_state gives embedding for each token.
        # https://github.com/huggingface/transformers/issues/7540
        if self.metric == "cosine":
            prototype_distances = -F.cosine_similarity(
                embedding, self.protolayer, dim=-1
            )
        elif self.metric == "L2":
            prototype_distances = -nes_torch(
                embedding, self.protolayer, dim=-1
            )
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
