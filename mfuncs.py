#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = r"""
BioCreative VI - Track 5 (CHEMPROT).

Main functions.

Information
-----------
Filename:
    mfuncs.py

Created:
    October 4, 2018

Last updated:
    October 17, 2018

Python version:
    3.6.4

"""

# third-party modules
import copy
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import normalize


def tokenize(s, vocabulary):
    r"""
    Tokenizes a string according to the available vocabulary.

    Parameters
    ----------
    s : str
        String to tokenize.
    vocabulary : str
        A `set` vocabulary with the available words.

    Returns
    -------
    tokens : list
        Tokens `list`.

    Example
    -------
    >>> s = 'this is a simple_string'
    >>> vocabulary = {'simple', 'string'}
    >>> tokenize(s, vocabulary)
    ['simple', 'string']
    >>>

    """
    s = s.replace('_', ' ')
    return [t for t in simple_preprocess(s, deacc=True) if t in vocabulary]


def normalized_sum(embeddings, size, dtype):
    r"""
    It returns the normalized sum of numpy vectors. If there are no
    vectors to sum, a zeros-vector is returned.

    Parameters
    ----------
    embeddings : list of numpy.ndarray (1D)
        Vectors to sum.
    size : int
        Vector length.
    dtype : str or numpy.dtype
        Numpy data type.

    Returns
    -------
    vec : numpy.ndarray (1D)
        Normalized sum.

    Example
    -------
    >>> embeddings = [[1, 2, 3], [4, 5, 6]]
    >>> size = 3
    >>> dtype = 'float32'
    >>> normalized_sum(embeddings, size, dtype)
    array([0.40160966, 0.56225353, 0.7228974 ], dtype=float32)
    >>> 

    """
    if len(embeddings) > 0:
        return normalize(
            np.sum(embeddings, axis=0, dtype=dtype).reshape((1, size))
        )[0]
    else:
        return np.zeros(size, dtype=dtype)


def normalized_rand(size, dtype):
    r"""
    It returns a normalized random numpy vector.

    Parameters
    ----------
    size : int
        Vector length.
    dtype : str or numpy.dtype
        Numpy data type.

    Returns
    -------
    vec : numpy.ndarray (1D)
        Normalized random vector.

    Example
    -------
    >>> v = normalized_rand(size=100, dtype='float32')
    >>>

    """
    return normalize(
        np.array(np.random.rand(size) - 0.5, dtype=dtype).reshape((1, size))
    )[0]


def tokseqs2intseqs(tokseqs, tok2int, **kwargs):
    r"""
    Convert token sequences to integer sequences.

    It makes use of the `pad_sequences` Keras function.

    Parameters
    ----------
    tokseqs : list of lists of str
        A `list` of sequences. Each sequence is a `list` of `str`
        tokens.
    tok2int : dict
        Mapping between a `str` token and the respective `int` index.
    **kwargs : dict, optional
        Keyword arguments that will be used as input for the
        `pad_sequences` function: `maxlen`, `padding` and `truncating`.

    Returns
    -------
    intseqs : numpy.ndarray (2D)
        Array with sequences of integers.

    Example
    -------
    >>> tokseqs = [
    ...     ['a'],
    ...     ['b', 'a'],
    ...     ['c', 'b', 'a'],
    ... ]
    >>> tok2int = {'a': 1, 'b': 2, 'c': 3}
    >>> intseqs = tokseqs2intseqs(tokseqs, tok2int, maxlen=2,
    ...     padding='pre', truncating='pre')
    >>> intseqs
    array([[0, 1],
           [2, 1],
           [2, 1]], dtype=int32)
    >>>

    """
    # deep copy (to not modify the input)
    tokseqs = copy.deepcopy(tokseqs)
    # replace tokens by indexes
    for i, tokseq in enumerate(tokseqs):
        for j, token in enumerate(tokseq):
            tokseqs[i][j] = tok2int[token]
    # pad_sequences
    intseqs = pad_sequences(sequences=tokseqs, **kwargs)
    return intseqs


def to_uncategorical(predictions, dtype='int32'):
    r"""
    Converts a matrix with probabilities to a class vector (integers).

    Parameters
    ----------
    predictions : numpy.ndarray (2D)
        Matrix with probabilities.
    dtype : str or numpy.dtype
        Numpy data type.

    Returns
    -------
    predicions_int : numpy.ndarray (1D)
        Class vector (integers).

    Example
    -------
    >>> pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.5, 0.4]])
    >>> to_uncategorical(pred)
    array([2, 1], dtype=int32)
    >>>

    """
    if len(predictions) > 0:
        return np.array(np.argmax(predictions, axis=1), dtype=dtype)
    else:
        return np.array(predictions, dtype=dtype)


def load_keyedvectors(fpath):
    r"""
    This function loads a `gensim.models.KeyedVectors` model.

    The `init_sims` method is internally called for providing vectors
    with a unit L2 norm _[1].

    Parameters
    ----------
    fpath : str
        Model filepath.

    Returns
    -------
    w2v : dict
        A `dict` mapping words to the respective embedding vectors.

    References
    ----------
    .. [1] https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors.init_sims

    """
    wv = KeyedVectors.load(fpath)
    wv.init_sims()
    return {w: v for w, v in zip(wv.index2word, wv.syn0norm)}


def load_word2vec(fpath):
    r"""
    This function loads a `gensim.models.Word2Vec` model.

    The `init_sims` method is internally called for providing vectors
    with a unit L2 norm _[1].

    Parameters
    ----------
    fpath : str
        Model filepath.

    Returns
    -------
    w2v : dict
        A `dict` mapping words to the respective embedding vectors.

    References
    ----------
    .. [1] https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.init_sims

    """
    w2v = Word2Vec.load(fpath)
    w2v.wv.init_sims()
    return {w: v for w, v in zip(w2v.wv.index2word, w2v.wv.syn0norm)}
