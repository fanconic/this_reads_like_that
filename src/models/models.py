from torch import nn
from transformers import (GPT2LMHeadModel,
                          GPT2ForSequenceClassification,
                          BertForSequenceClassification)


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
            'gpt2', num_labels=num_class)
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
            'bert-base-uncased', num_labels=num_class)
        # self.gpt2_classifier.config.pad_token_id = 50256
        if model_configs["freeze_layers"]:
            for param in self.bert_classifier.base_model.parameters():
                param.requires_grad = False

    def forward(self, tokenized_text, attention_mask):
        # print(tokenized_text.shape)
        return self.bert_classifier(tokenized_text)
