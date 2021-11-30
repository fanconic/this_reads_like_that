from torch import nn
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification


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

        self.gpt2 = GPT2ForSequenceClassification.from_pretrained('gpt2')
        # self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, tokenized_text, offsets):
        # embedded = self.gpt2(text)
        return self.gpt2(tokenized_text)[0]