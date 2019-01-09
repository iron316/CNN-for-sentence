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

def main():
    train_dirs = ['natsume','edogawa']
    test_dirs = ['test_natsume','test_edogawa']

    train_data = create_dataset(train_dirs)
    test_data = create_dataset(test_dirs)
    print("train  : {}".format(len(train_data)))
    print("test   : {}".format(len(test_data)))
    with open('sentence_vec.pickle', 'wb') as wbf:
        pickle.dump(train_data, wbf)
    with open('test_sentence_vec.pickle','wb') as wbf:
        pickle.dump(test_data,wbf)


if __name__ == '__main__':
    main()
