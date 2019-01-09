
# coding: utf-8

# In[1]:


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
    sentences = [[model[word]
                  for word in words if word in model.vocab]for words in wakati]
    sentence_vec = []
    for sentence in sentences:
        pad_sentence = np.zeros((100, 200))
        sentence = np.asarray(sentence).reshape((-1, 200))
        if 10 < sentence.shape[0] < 100:
            pad_sentence[:len(sentence)] = sentence
            sentence_vec.append(pad_sentence.astype(np.float32))
        else:
            continue
    return sentence_vec


def trans_vector(files, model):
    dir_vectors = []
    for file_name in files:
        with open(file_name, "rb") as rf:
            bindata = rf.read()
            text = bindata.decode('shift_jis')
            lines = text_split(text)
            wakati = trans_wakati(lines)
            sentence_vec = trans_array(wakati, model)
        dir_vectors.extend(sentence_vec)
    return dir_vectors
