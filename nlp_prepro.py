
# coding: utf-8

from gensim.models import KeyedVectors
import zipfile
import re
import MeCab
import numpy as np


def text_split(text):
    text = re.split(r'\-{5,}', text)[2]
    text = re.split(r'底本：', text)[0]
    text = re.sub(r'《.+?》', '', text)
    text = re.sub(r'［＃.+?］', '', text)
    text = text.strip()
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\u3000", "")
    lines = text.split("。")
    return lines


def trans_wakati(lines):
    t = MeCab.Tagger('-Owakati')
    t.parse('')
    wakati_list = []
    for line in lines:
        node = t.parseToNode(line)
        surface_list = []
        while node:
            surface_list.append(node.surface)
            node = node.next
        wakati_list.append(surface_list)
    return wakati_list


def trans_array(wakati, model):
    sentence = [model[word] if word in model.vocab else np.random.normal(size=(200))
                for word in wakati]
    if len(sentence) > 250:
        sentence = sentence[:250]
    sentence_vec = np.asarray(sentence).reshape(-1, 200)
    return sentence_vec


def to_idx(wakati, dct):
    idx_list = [dct.doc2idx(sentence, unknown_word_index=-2)
                for sentence in wakati]
    return idx_list


def split_sentence(files, dct):
    dir_sentence = []
    for file_name in files:
        with open(file_name, "rb") as rbf:
            bindata = rbf.read()
            text = bindata.decode('shift_jis')
            lines = text_split(text)
            wakati = trans_wakati(lines)
            idx = to_idx(wakati, dct)
        dir_sentence.extend(idx)
    return dir_sentence


def trans_vector(files, dct):
    dir_vectors = []
    for file_name in files:
        with open(file_name, "rb") as rf:
            bindata = rf.read()
            text = bindata.decode('shift_jis')
            lines = text_split(text)
            wakati = trans_wakati(lines)
            sentence_vec = trans_array(wakati, model)
            print(file_name)
        print("{} vector : {}".format(file_name, len(sentence_vec)))
        dir_vectors.extend(sentence_vec)
    return dir_vectors
