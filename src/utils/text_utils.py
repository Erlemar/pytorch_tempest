import pickle
from collections import Counter
from typing import List, Union, Dict, Tuple, Any, Optional

import numpy as np
import numpy.typing as npt
import torch.nn as nn
from gensim.models import FastText
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj


def _generate_tag_to_idx(cfg: DictConfig, entities_names: List) -> dict:
    """
    Generate tag-idx vocab
    Args:
        cfg: config
        entities_names: list of train entities
    Returns:
        tag-idx vocab
    """
    tag_to_idx: Dict = {}
    for ind, entity in enumerate(entities_names):
        if cfg.datamodule.params.use_bulio_tokens:
            tag_to_idx[f'B-{entity}'] = len(tag_to_idx)
            tag_to_idx[f'I-{entity}'] = len(tag_to_idx)
            tag_to_idx[f'L-{entity}'] = len(tag_to_idx)
            tag_to_idx[f'U-{entity}'] = len(tag_to_idx)
        else:
            tag_to_idx[entity] = ind + 1
    for special_tag in ['O', 'PAD']:
        tag_to_idx[special_tag] = len(tag_to_idx)

    return tag_to_idx


def _generate_word_to_idx(
    data: List,
    use_pad_token: bool = False,
    use_unk_token: bool = False,
    min_words: Union[int, float] = 0.0,
    max_words: Union[int, float] = 1.0,
) -> Dict[str, int]:
    """
    Generate word-idx vocab
    Args:
        data: ner dataset
        use_pad_token: pad_token flag
        use_pad_token: pad_token flag
        min_words: min word count
        max_words: max word count
    Returns:
        word-idx vocab
    """
    all_words = []
    if use_pad_token:
        all_words.append('<pad>')
    if use_unk_token:
        all_words.append('<unk>')

    for tokens in data:
        all_words.extend(tokens['text'])
    count = Counter(all_words).most_common()
    max_count = count[0][1]
    if isinstance(min_words, float):
        min_words = max_count * min_words
    if isinstance(max_words, float):
        max_words = max_count * max_words

    all_words = [w[0] for w in count if max_words >= w[1] >= min_words]
    # all_words = ['<pad>', '<unk>'] + all_words
    word_to_idx = {k: v for k, v in zip(all_words, range(0, len(all_words)))}

    return word_to_idx


def get_vectorizer(cfg: DictConfig, word_to_idx: Dict) -> nn.Module:
    """
    Get model

    Args:
        word_to_idx:
        cfg: config

    Returns:
         initialized model
    """

    vectorizer = load_obj(cfg.datamodule.vectorizer_class_name)
    vectorizer = vectorizer(
        word_to_idx=word_to_idx,
        embeddings_path=cfg.datamodule.embeddings_path,
        embeddings_type=cfg.datamodule.embeddings_type,
        embeddings_dim=cfg.datamodule.embeddings_dim,
    )

    return vectorizer


def get_word_to_idx(datasets: List) -> Dict[str, int]:
    """
    Get dictionary with words and indexes
    Args:
        datasets:

    Returns:
        Dict with words and indexes
    """
    word_to_idx: Dict[str, int] = {'пропущено': 1}
    for dataset in datasets:
        for claim in dataset:
            for word in claim['text']:
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx) + 1

    return word_to_idx


def get_coefs(word: str, *arr: npt.ArrayLike) -> Tuple[str, npt.ArrayLike]:
    """
    Get word and coefficient from line in embeddings
    Args:
        word:
        *arr:

    Returns:
        word and array with values
    """
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(embedding_path: str, embedding_type: str = 'fasttext') -> Union[Dict, Any]:
    """
    Load embeddings into dictionary
    Args:
        embedding_path: path to embeddings
        embedding_type: type of pretrained embeddings ('word2vec', 'glove', 'fasttext')
    Returns:
        loaded embeddings
    """
    if 'pkl' in embedding_path:
        with open(embedding_path, 'rb') as vec_f:
            return pickle.load(vec_f)
    elif embedding_type == 'word2vec':
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f)
            return dict(get_coefs(*line.strip().split(' ')) for line in f)
    elif embedding_type == 'glove':
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
            return dict(get_coefs(*line.strip().split(' ')) for line in f)
    elif embedding_type == 'fasttext':
        return FastText.load(embedding_path)
    else:
        return None


def get_vector(embedding_type: str, embedding_index: dict, word: str) -> Optional[npt.ArrayLike]:
    """
    Return vector in relation to embedding_type parameter
    Args:
        embedding_type: type of pretrained embeddings ('word2vec', 'glove', 'fasttext')
        embedding_index: dict of vectors
        word: keyword
    Returns:
        word vector
    """
    emb = None

    if embedding_type in ['word2vec', 'glove']:
        emb = embedding_index.get(word)
    elif embedding_type == 'fasttext' and type(embedding_index) == FastText:
        emb = embedding_index[word]
    elif embedding_type == 'fasttext' and type(embedding_index) == dict:
        emb = embedding_index.get(word)
    return emb


def build_matrix(
    word_dict: Dict,
    embedding_path: str = '',
    embeddings_type: str = 'fasttext',
    max_features: int = 100000,
    embed_size: int = 300,
) -> Tuple[npt.ArrayLike, int, List]:
    """
    Create embedding matrix

    Args:
        embedding_path: path to embeddings
        embeddings_type: type of pretrained embeddings ('word2vec', 'glove', 'fasttext')
        word_dict: tokenizer
        max_features: max features to use
        embed_size: size of embeddings

    Returns:
        embedding matrix, number of of words and the list of not found words
    """
    if embeddings_type not in ['word2vec', 'glove', 'fasttext']:
        raise ValueError('Unacceptable embedding type.\nPermissible values: word2vec, glove, fasttext')
    embedding_index = load_embeddings(embedding_path, embeddings_type)
    nb_words = min(max_features, len(word_dict))
    if embeddings_type in ['word2vec', 'glove']:
        embed_size = embed_size if embed_size is not None else len(list(embedding_index.values())[0])
        all_embs = np.stack(embedding_index.values())  # type: ignore
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    elif embeddings_type == 'fasttext':
        embedding_matrix = np.zeros((nb_words, embed_size))

    unknown_words = []
    for word, i in word_dict.items():
        key = word
        embedding_vector = get_vector(embeddings_type, embedding_index, word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = get_vector(embeddings_type, embedding_index, word.lower())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = get_vector(embeddings_type, embedding_index, word.upper())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        embedding_vector = get_vector(embeddings_type, embedding_index, word.capitalize())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        unknown_words.append(key)
    return embedding_matrix, nb_words, unknown_words


def pad_sequences(
    sequences: List,
    maxlen: Optional[int],
    dtype: str = 'int32',
    padding: str = 'post',
    truncating: str = 'post',
    value: int = 0,
) -> npt.ArrayLike:
    """Pad sequences to the same length.
    from Keras

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. ' 'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape: Tuple[int, ...] = ()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" ' 'not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f'Shape of sample {trunc.shape[1:]} of sequence at position {idx}'
                f'is different from expected shape {sample_shape}'
            )

        if padding == 'post':
            x[idx, : len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x
