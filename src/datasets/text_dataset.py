from typing import List, Dict, Tuple

import numpy as np
import numpy.typing as npt
import torch
from gensim.models import FastText
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from src.utils.text_utils import pad_sequences, build_matrix


class NerDataset(Dataset):
    def __init__(self, ner_data: List, word_to_idx: Dict, cfg: DictConfig, tag_to_idx: Dict, preload: bool = True):
        """
        Prepare data for wheat competition.
        Args:
            ner_data: data
            word_to_idx: mapping of words do indexes
            cfg: config with parameters
            tag_to_idx: mapping of tags do indexes
        """
        # self.ner_data = ner_data
        self.data_len = len(ner_data)
        self.cfg = cfg
        self.preload = preload
        if preload:
            self.tokens = np.array(
                [[word_to_idx[w] if w in word_to_idx.keys() else 1 for w in line['text']] for line in ner_data]
            )
            self.labels = np.array([[tag_to_idx[w] for w in line['labels']] for line in ner_data])
        else:
            self.ner_data = ner_data
            self.cfg = cfg
            self.word_to_idx = word_to_idx
            self.tag_to_idx = tag_to_idx

    def __getitem__(self, idx: int) -> Tuple[npt.ArrayLike, int, npt.ArrayLike]:

        if self.preload:
            return self.tokens[idx], len(self.tokens[idx]), self.labels[idx]

        else:
            line = self.ner_data[idx]
            tokens = [self.word_to_idx[w] if w in self.word_to_idx.keys() else 1 for w in line['text']]
            if self.cfg.dataset.params.use_bulio_tokens:
                labels = [self.tag_to_idx[w] for w in line['labels']]
            else:
                labels = [self.tag_to_idx[w] for w in line['labels_flat']]
            return np.array(tokens), len(tokens), np.array(labels)

    def __len__(self) -> int:
        return self.data_len


class Collator:
    def __init__(self, test=False, percentile=100, pad_value=0):
        self.test = test
        self.percentile = percentile
        self.pad_value = pad_value

    def __call__(self, batch):
        tokens, lens, labels = zip(*batch)
        lens = np.array(lens)

        max_len = min(int(np.percentile(lens, self.percentile)), 100)

        tokens = torch.tensor(
            pad_sequences(tokens, maxlen=max_len, padding='post', value=self.pad_value), dtype=torch.long
        )
        lens = torch.tensor([min(i, max_len) for i in lens], dtype=torch.long)
        labels = torch.tensor(
            pad_sequences(labels, maxlen=max_len, padding='post', value=self.pad_value), dtype=torch.long
        )

        return tokens, lens, labels


class Vectorizer(nn.Module):
    """
    Transform tokens to embeddings
    """

    def __init__(self, word_to_idx: Dict, embeddings_path: str, embeddings_type: str, embeddings_dim: int = 100):
        super(Vectorizer, self).__init__()
        self.weights_matrix, _, _ = build_matrix(
            word_to_idx, embeddings_path, embeddings_type, max_features=len(word_to_idx), embed_size=embeddings_dim
        )
        self.weights_matrix = torch.tensor(self.weights_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(self.weights_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        embed = self.embedding(x)
        return embed


class InferenceVectorizer:
    """
    Transform tokens to embeddings
    """

    def __init__(self, embeddings_path: str):
        self.fasttext = FastText.load(embeddings_path)

    def __call__(self, claim):
        splited_claim = claim.split()
        with torch.no_grad():
            data_tensor = torch.tensor([self.fasttext[token] for token in splited_claim]).unsqueeze(0)
            length_tensor = torch.tensor(len(splited_claim)).unsqueeze(0)

        return data_tensor, length_tensor
