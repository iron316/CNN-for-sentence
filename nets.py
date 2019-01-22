import chainer.links as L
import chainer.functions as F
import chainer
from chainer.dataset.convert import concat_examples
from chainer import reporter


class TextClassifier(chainer.Chain):
    def __init__(self, encoder):
        super(TextClassifier, self).__init__()
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

    def predict(self, xs):
        out_put = self.encoder(xs)
        return out_put


class NonStatic(chainer.Chain):
    def __init__(self, w2v_w, batch_size):
        self.w2v_w = w2v_w
        self.batch = batch_size
        super(NonStatic, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1],
                ignore_label=-1, initialW=self.w2v_w)
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, xs):
        x = concat_examples(xs, padding=-1)
        len_x = len(x[0])
        h = self.embed(x)
        sentence_vec = F.reshape(h, (-1, 1, len_x, 200))
        h_3 = F.max(
            F.tanh(self.cnn_w3(sentence_vec)), axis=1)
        h_4 = F.max(
            F.tanh(self.cnn_w4(sentence_vec)), axis=1)
        h_5 = F.max(
            F.tanh(self.cnn_w5(sentence_vec)), axis=1)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y


class Static(chainer.Chain):
    def __init__(self, w2v_w, batch_size):
        self.w2v_w = w2v_w
        self.batch = batch_size
        super(Static, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1],
                ignore_label=-1, initialW=self.w2v_w)
            self.embed.disable_update()
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, xs):
        x = concat_examples(xs, padding=-1)
        len_x = len(x[0])
        h = self.embed(x)
        sentence_vec = F.reshape(h, (-1, 1, len_x, 200))
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


class TwoChannel(chainer.Chain):
    def __init__(self, w2v_w, batch_size):
        self.w2v_w = w2v_w
        self.batch = batch_size
        super(TwoChannel, self).__init__()
        with self.init_scope():
            self.embed1 = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1],
                ignore_label=-1, initialW=self.w2v_w)
            self.embed1.disable_update()
            self.embed2 = L.EmbedID(
                self.w2v_w.shape[0], self.w2v_w.shape[1],
                ignore_label=-1, initialW=self.w2v_w)
            self.cnn_w3 = L.Convolution2D(None, 100, (3, 200))
            self.cnn_w4 = L.Convolution2D(None, 100, (4, 200))
            self.cnn_w5 = L.Convolution2D(None, 100, (5, 200))
            self.fc = L.Linear(None, 5)

    def __call__(self, xs):
        x = concat_examples(xs, padding=-1)
        len_x = len(x[0])
        h1 = F.reshape(self.embed1(x), (-1, 1, len_x, 200))
        h2 = F.reshape(self.embed2(x), (-1, 1, len_x, 200))
        sentence_vec = F.concat([h1, h2], axis=1)
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
