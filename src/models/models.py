from torch import nn


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
