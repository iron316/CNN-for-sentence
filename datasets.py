# coding: utf-8

import glob
from nlp_prepro import trans_vector
from gensim.models import KeyedVectors
import pickle

def create_dataset(dirs):
    model_dir = "entity_vector.model.bin"
    model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    datasets = []
    for label, dir_name in enumerate(dirs):
        files = glob.glob(dir_name+"/*.txt")
        tokens = trans_vector(files, model)
        datasets.extend([(token, label) for token in tokens])
    return datasets

def __main__():
    dirs = ['natsume','edogawa']
    data = create_dataset(dirs)
    print(len(data))
    #with open('sentence_vec.pickle', 'wb') as wbf:
    #    pickle.dump(data, wbf)

if __name__ == '__main__':
    __main__()
