import chainer.links as L
import chainer.functions as F
import chainer
from chainer.dataset.convert import concat_examples
import numpy as np
from chainer import reporter


class text_classifier(chainer.Chain):
    def __init__(self, encoder):
        super(text_classifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder

    def forward(self, xs, ys):
        concat_output = self.encoder(xs)
        concat_true = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_output, concat_true)
        accuracy = F.accuracy(concat_output, concat_true)
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        concat_encodings = F.dropout(self.encoder(xs), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).array
        elif argmax:
            return self.xp.argmax(concat_outputs.array, axis=1)



class non_static(chainer.Chain):
    def __init__(self, w2v_w, batch_size):
        self.w2v_w = w2v_w
        self.batch = batch_size
        super(non_static, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1], ignore_label=-5, initialW=self.w2v_w)
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, xs):
        x = concat_examples(xs, padding=-5)
        h = self.embed(x)
        sentence_vec = F.dropout(
            F.reshape(h, (self.batch, 1, -1, 200)), ratio=0.2)
        h_3 = F.max(
            F.tanh(self.cnn_w3(sentence_vec)), axis=2)
        h_4 = F.max(
            F.tanh(self.cnn_w4(sentence_vec)), axis=2)
        h_5 = F.max(
            F.tanh(self.cnn_w5(sentence_vec)), axis=2)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y


class static(chainer.Chain):
    def __init__(self, w2v_w, batch_size):
        self.w2v_w = w2v_w
        self.batch = batch_size
        super(static, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1], ignore_label=-5, initialW=self.w2v_w)
            self.embed.disable_update()
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, xs):
        x = concat_examples(xs, padding=-5)
        h = self.embed(x)
        sentence_vec = F.dropout(
            F.reshape(h, (self.batch, 1, -1, 200)), ratio=0.2)
        h_3 = F.max_pooling_2d(
            F.tanh(self.cnn_w3(sentence_vec)),sentence_vec.shape[2] )
        h_4 = F.max_pooling_2d(
            F.tanh(self.cnn_w4(sentence_vec)), sentence_vec.shape[2])
        h_5 = F.max_pooling_2d(
            F.tanh(self.cnn_w5(sentence_vec)), sentence_vec.shape[2])
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y
