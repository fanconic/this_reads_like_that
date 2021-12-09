import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import GPT2Tokenizer, BertTokenizer

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    """Yielding function for the text in the data corpora.
    Args:
        data_iter: data iterator
    Yields:
        documents in the corpora
    """
    for _, text in data_iter:
        yield tokenizer(text)


def build_vocab(data_iter):
    """Build vocabulary
    Args:
        data_iter: either train or test
    Returns:
        vocabulary

    """
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def build_loader(data_iter, batch_size=8, device="cpu", vocab=None, config=None):
    """Build data loader
    Args:
        data_iter: either train or test
        batch_size (default 8): batch size
        device (default "cpu"): device that the tensors are on
    Returns:
        data loader & vocabulary

    """
    if vocab == None:
        vocab = build_vocab(data_iter)

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def label_pipeline(x):
        return int(x) - 1

    def collate_batch(batch):
        """Collate function to generate batches from the dataset
        Args:
            batch: current batch from the iterator
        returns:
            labels, text, offset
        """
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    def collate_batch_lm(batch):
        """Collate function to generate batches from the dataset
        Args:
            batch: current batch from the iterator
        returns:
            labels, text, offset
        """
        label_list, text_list, offsets = [], [], [0]
        if "gpt2" in config["model"]["submodel"]:
            tkizr = GPT2Tokenizer.from_pretrained("gpt2", return_tensors="pt")
            tkizr.pad_token = "[PAD]"
        elif "bert" in config["model"]["submodel"]:
            tkizr = BertTokenizer.from_pretrained(
                "bert-base-uncased", return_tensors="pt"
            )
        text = list(list(zip(*batch))[1])  # 0: label, 1: text
        tokenized_text = tkizr(text, return_tensors="pt", padding=True)

        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        return (
            label_list.to(device),
            tokenized_text["input_ids"].to(device),
            tokenized_text["attention_mask"].to(device),
        )

    use_collate_lm = (
        "gpt2" in config["model"]["submodel"] or "bert" in config["model"]["submodel"]
    )
    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch if not use_collate_lm else collate_batch_lm,
    )

    return dataloader, vocab
