# coding: utf-8

import glob
from nlp_prepro import split_sentence
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
import pickle
import numpy as np


def create_dataset(dirs, model):
    i2w = model.index2word
    dct = Dictionary([i2w])
    datasets = []
    for i, dir_name in enumerate(dirs):
        files = glob.glob(dir_name+"/*.txt")
        tokens = split_sentence(files, dct)
        labels = np.array([i]*len(tokens)).astype(np.int32)
        datasets.extend([(token, label)
                         for token, label in zip(tokens, labels)])

    return datasets
