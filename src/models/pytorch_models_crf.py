from typing import Tuple, Dict

import torch
from torch import nn
from torchcrf import CRF

from src.models.layers.layers import SpatialDropout


class BiLSTMCRF(nn.Module):
    """
    New model without nn.Embedding layer
    """

    def __init__(
        self, tag_to_idx: Dict, embeddings_dim: int = 100, hidden_dim: int = 4, spatial_dropout: float = 0.2
    ):  # type: ignore
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embeddings_dim
        self.hidden_dim = hidden_dim
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx.values())
        self.crf = CRF(self.tagset_size, batch_first=True)
        self.embedding_dropout = SpatialDropout(spatial_dropout)

        self.lstm = nn.LSTM(embeddings_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hidden2tag2 = nn.Linear(hidden_dim // 2, self.tagset_size)

    def _get_lstm_features(self, embeds: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        LSTM forward

        Args:
            embeds: batch with embeddings
            lens: lengths of sequences
        """
        embeds = self.embedding_dropout(embeds)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, self.hidden = self.lstm(packed_embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag2(self.hidden2tag(lstm_out.reshape(embeds.shape[0], -1, self.hidden_dim)))
        return lstm_feats

    def forward(
        self, embeds: torch.Tensor, lens: torch.Tensor, tags: torch.Tensor = None
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Forward

        Args:
            embeds: batch with embeddings
            lens: lengths of sequences
            tags: list of tags (optional)
        """
        lstm_feats = self._get_lstm_features(embeds, lens)
        if tags is not None:
            mask = tags != self.tag_to_idx['PAD']
            loss = self.crf(lstm_feats, tags, mask=mask)
        else:
            loss = 0
        if tags is not None:
            mask = torch.tensor(mask)
            tag_seq = self.crf.decode(emissions=lstm_feats, mask=mask)
        else:
            tag_seq = self.crf.decode(lstm_feats)
        score = 0
        tag_seq = torch.tensor([i for j in tag_seq for i in j]).type_as(embeds)

        return score, tag_seq, -loss
