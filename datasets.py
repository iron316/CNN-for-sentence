# coding: utf-8

import glob
from nlp_prepro import split_sentence
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
import pickle
import numpy as np


def create_dataset(dirs, model):
    i2w = model.index2word
    n_vocab = len(i2w)
    dct = Dictionary([i2w])
    datasets = []
    for i, dir_name in enumerate(dirs):
        files = glob.glob(dir_name+"/*.txt")
        tokens = split_sentence(files, dct, n_vocab)
        labels = np.array([i]*len(tokens)).astype(np.int32)
        datasets.extend([(token, np.array([label]))
                         for token, label in zip(tokens, labels)])

    return datasets
