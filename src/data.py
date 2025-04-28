import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Iterable, Tuple
from pyvi import ViTokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

import logging
logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Tokenizes the input text into a list of words."""
    # TODO: build a custom tokenizer
    return ViTokenizer.tokenize(text).split()

def yield_tokens(data_iter: Iterable[Tuple[str, str]]):
    """Generator that feeds torchtext’s vocab builder."""
    for text in data_iter:  # add _, text if ignore_idex=False
        yield tokenize(text)

class CustomDataset(Dataset):
    def __init__(self, data: List, labels: List):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class CustomDataLoader(DataLoader):
    def __init__(self, 
                 dataset: Dataset, 
                 vocab,
                 batch_size: int, 
                 shuffle: bool = True,
                 cfg = None):
        
        super().__init__(dataset, 
                         batch_size=batch_size, 
                         shuffle=shuffle,
                         collate_fn=self.collate_fn)
        self.cfg = cfg
        self.vocab = vocab
        
    def _encode(self, label, text: str) -> Tuple[List[int], List[int]]:
        toks = tokenize(text)[:self.cfg.max_len]
        ids = [self.vocab[token] for token in toks]
        padding_len = self.cfg.max_len - len(ids)
        ids.extend([self.vocab["<pad>"]] * padding_len)
        # cast label to float32
        return torch.tensor(ids), torch.tensor(label, dtype=torch.float32)
    
    def collate_fn(self, batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*[self._encode(label, text) for text, label in batch])     # each batch (B, L), (B, 1)
        logger.info(f"xs len: {len(xs)}, ys len: {len(ys)}")
        return torch.stack(xs), torch.stack(ys)

def build_loaders(cfg):
    """Builds the data loaders for training and validation."""
    train_data_path = cfg.train.data_path
    val_data_path = cfg.val.data_path
    
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    full_df = pd.concat([train_df, val_df], ignore_index=True)  # TODO: add test df

    logger.info(f"Loading training and validation data...")   # TODO: alter print by logger
    train_dataset = CustomDataset(train_df["sentence"], train_df["label"])
    val_dataset = CustomDataset(val_df["sentence"], val_df["label"])

    vocab = build_vocab_from_iterator(yield_tokens(full_df["sentence"]), specials=cfg.specials)  # <pad>: 0, <unk>: 1 
    vocab.set_default_index(vocab[cfg.specials[1]])    # <unk>

    train_loader = CustomDataLoader(train_dataset, vocab, cfg.batch_size, shuffle=True, cfg=cfg)
    val_loader = CustomDataLoader(val_dataset, vocab, cfg.batch_size, shuffle=False, cfg=cfg)

    return train_loader, val_loader, vocab

def build_test_loader(cfg):
    """Builds the data loader for testing."""
    train_data_path = cfg.train.data_path
    val_data_path = cfg.val.data_path
    test_data_path = cfg.test.data_path
    
    logger.info(f"Building vocab...")
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    full_df = pd.concat([train_df, val_df], ignore_index=True)  # TODO: add test df
    vocab = build_vocab_from_iterator(yield_tokens(full_df["sentence"]), specials=cfg.specials)  # or 
    vocab.set_default_index(vocab[cfg.specials[1]])    # <unk>
    
    logger.info(f"Loading test data...")
    test_df = pd.read_csv(test_data_path)
    test_dataset = CustomDataset(test_df["sentence"], test_df["label"])

    test_loader = CustomDataLoader(test_dataset, vocab, cfg.batch_size, shuffle=False, cfg=cfg)

    return test_loader, vocab