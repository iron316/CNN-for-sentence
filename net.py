import chainer.links as L
import chainer.functions as F
import chainer
from chainer.dataset.convert import concat_examples
import numpy as np


class non_static(chainer.Chain):
    def __init__(self, w2v_w):
        self.w2v_w = w2v_w
        super(non_static, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1], initialW=self.w2v_w)
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, x):
        sentence_vec = idx_to_vec(self.embed, x)
        h_3 = F.max_pooling_2d(
                F.tanh(self.cnn_w3(sentence_vec)), 100)
        h_4 = F.max_pooling_2d(
                F.tanh(self.cnn_w4(sentence_vec)), 100)
        h_5 = F.max_pooling_2d(
                F.tanh(self.cnn_w5(sentence_vec)), 100)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y


def idx_to_vec(embed, x):
    e = embed(x)
    e = F.dropout(e, ratio=0.2)
    e = F.transpose(e, (0, 1, 2))
    e = F.reshape(e, (-1, 1, 100, 200))
    # e = e.reshape(-1, 1, 100, 200)
    return e


class static(chainer.Chain):
    def __init__(self, w2v_w):
        self.w2v_w = w2v_w
        super(non_static, self).__init__()
        with self.init_scope():
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, x):
        sentence_vec = F.reshape(F.embed_id(
            x, self.w2v_w.astype(np.float32)), (200, 1, -1, 200))
        sentence_vec = F.dropout(sentence_vec, ratio=0.2)
        h_3 = F.max_pooling_2d(
             F.tanh(self.cnn_w3(sentence_vec)), 100)
        h_4 = F.max_pooling_2d(
             F.tanh(self.cnn_w4(sentence_vec)), 100)
        h_5 = F.max_pooling_2d(
             F.tanh(self.cnn_w5(sentence_vec)), 100)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y
