import json
from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.datasets.text_dataset import Collator
from src.utils.technical_utils import load_obj
from src.utils.text_utils import _generate_tag_to_idx, _generate_word_to_idx, get_vectorizer


class NerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    @staticmethod
    def load_sentences(filepath):
        """
        Load sentences (separated by newlines) from dataset

        Parameters
        ----------
        filepath : str
            path to corpus file

        Returns
        -------
        List of sentences represented as dictionaries

        """

        sentences, tok, ne = [], [], []

        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line in [('-DOCSTART- -X- -X- O\n'), '\n']:
                    # Sentence as a sequence of tokens, POS, chunk and NE tags
                    if tok != []:
                        sentence = {'text': [], 'labels': []}
                        sentence['text'] = tok
                        sentence['labels'] = ne

                        # Once a sentence is processed append it to the list of sentences
                        sentences.append(sentence)

                    # Reset sentence information
                    tok = []
                    ne = []
                else:
                    splitted_line = line.split(' ')

                    # Append info for next word
                    tok.append(splitted_line[0])
                    ne.append(splitted_line[3].strip('\n'))

        return sentences

    def setup(self, stage=None):
        # with open(f'{self.cfg.datamodule.folder_path}{self.cfg.datamodule.file_name}', 'r', encoding='utf-8') as f:
        ner_data = self.load_sentences(f'{self.cfg.datamodule.folder_path}{self.cfg.datamodule.file_name}')

        # generate tag_to_idx
        labels = [labels['labels'] for labels in ner_data]
        flat_labels = list({label for sublist in labels for label in sublist})
        entities_names = sorted({label.split('-')[1] for label in flat_labels if label != 'O'})
        if self.cfg.datamodule.tag_to_idx_from_labels:
            self.tag_to_idx = {v: i for i, v in enumerate({i for j in labels for i in j}) if v != 'O'}
            for special_tag in ['O', 'PAD']:
                self.tag_to_idx[special_tag] = len(self.tag_to_idx)
        else:
            self.tag_to_idx = _generate_tag_to_idx(self.cfg, entities_names)

        # load or generate word_to_idx
        if self.cfg.datamodule.word_to_idx_name:
            with open(
                f'{self.cfg.datamodule.folder_path}{self.cfg.datamodule.word_to_idx_name}', 'r', encoding='utf-8'
            ) as f:
                self.word_to_idx = json.load(f)
        else:
            self.word_to_idx = _generate_word_to_idx(ner_data)

        train_data, valid_data = train_test_split(
            ner_data, random_state=self.cfg.training.seed, test_size=self.cfg.datamodule.valid_size
        )

        dataset_class = load_obj(self.cfg.datamodule.class_name)

        self.train_dataset = dataset_class(
            ner_data=train_data, cfg=self.cfg, word_to_idx=self.word_to_idx, tag_to_idx=self.tag_to_idx
        )
        self.valid_dataset = dataset_class(
            ner_data=valid_data, cfg=self.cfg, word_to_idx=self.word_to_idx, tag_to_idx=self.tag_to_idx
        )

        self._vectorizer = get_vectorizer(self.cfg, self.word_to_idx)
        self.collate = Collator(percentile=100, pad_value=self.tag_to_idx['PAD'])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

        return valid_loader

    def test_dataloader(self):
        return None
