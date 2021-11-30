import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import GPT2Tokenizer

tokenizer = get_tokenizer('basic_english')


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
    vocab = build_vocab_from_iterator(
        yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def build_loader(data_iter, batch_size=8, device="cpu", vocab=None, model_name=None):
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
        if model_name == "gpt2":
            tkizr = GPT2Tokenizer.from_pretrained(model_name, return_tensors="pt")
            return tkizr.encode_plus(x)
        else:
            return vocab(tokenizer(x))
    def label_pipeline(x): return int(x) - 1

    def collate_batch(batch):
        """ Collate function to generate batches from the dataset
        Args:
            batch: current batch from the iterator
        returns:
            labels, text, offset
        """
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(
                text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    dataloader = DataLoader(data_iter, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_batch)

    return dataloader, vocab
