from datasets import create_dataset
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import concat_examples
from chainer import iterators, training, optimizers
from net import non_static
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gensim.models import KeyedVectors
plt.switch_backend('agg')


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def get_w(model):
    w = model.vectors
    unk_vec = np.random.normal(size=(200)).astype(np.float32)
    zero_vec = np.zeros((200), dtype=np.float32)
    w2v_w = np.vstack((w, unk_vec))
    w2v_w = np.vstack((w2v_w, zero_vec))
    return w2v_w


def show_test_performance(model, test_iter, gpu_id, batch_size):
    if gpu_id >= 0:
        model.to_gpu()
    test_evaluator = training.extensions.Evaluator(
        test_iter, model, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])


class CNN_fsc(chainer.Chain):
    def __init__(self):
        super(CNN_fsc, self).__init__()
        with self.init_scope():
            self.cnn_w3 = L.Convolution2D(
                None, 100, ksize=(3, 200), pad=0)
            self.cnn_w4 = L.Convolution2D(
                None, 100, ksize=(4, 200), pad=0)
            self.cnn_w5 = L.Convolution2D(
                None, 100, ksize=(5, 200), pad=0)
            self.fc = L.Linear(None, 5)

    def __call__(self, x):
        h = x.reshape((-1, 1, 250, 200))
        wc_sentence = F.dropout(h, ratio=0.2)
        h_3 = F.max_pooling_2d(F.tanh(self.cnn_w3(wc_sentence)), 250)
        h_4 = F.max_pooling_2d(F.tanh(self.cnn_w4(wc_sentence)), 250)
        h_5 = F.max_pooling_2d(F.tanh(self.cnn_w5(wc_sentence)), 250)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y


class Preprocessdataset(chainer.dataset.DatasetMixin):

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        idx_list, label = self. values[i]
        pad = [-1]*100
        if len(idx_list) >= 100:
            idx_list = idx_list[:100]
        pad[:len(idx_list)] = idx_list
        return np.array(pad, dtype=np.int32), label


def main():
    reset_seed(0)
    model_dir = "entity_vector.model.bin"
    w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    train_dirs = ['natsume', 'edogawa' 'dazai', 'akutagawa', 'miyazawa']
    test_dirs = ['test_natsume', 'test_edogawa',
                 'test_dazai', 'test_akutagawa', 'test_miyazawa']
    train = create_dataset(train_dirs, w2v_model)
    valid = create_dataset(test_dirs, w2v_model)
    #test, train_val = split_dataset_random(data, 10, seed=19910927)
    # with open('sentence_vec.pickle', 'rb') as rbf:
    #    train = pickle.load(rbf)
    # with open('test_sentence_vec.pickle', 'rb') as rbf:
    #    valid = pickle.load(rbf)
    #train, valid = split_dataset_random(data, int(len(data) * 0.8), seed=19910927)

    batch_size = 128
    gpu_id = 0
    max_epoch = 100
    train = Preprocessdataset(train)
    valid = Preprocessdataset(valid)

    w2v_w = get_w(w2v_model)

    train_iter = iterators.MultithreadIterator(train, batch_size, n_threads=4)
    valid_iter = iterators.MultithreadIterator(
        valid, batch_size, n_threads=4, repeat=False, shuffle=False)
    # test_iter = iterators.SerialIterator(
    #    test, batch_size, repeat=False, shuffle=False)

    net = non_static(w2v_w)

    model = L.Classifier(net)

    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    optimizer = optimizers.Adam().setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="result")

    trainer.extend(training.extensions.LogReport())
    trainer.extend(training.extensions.Evaluator(
        valid_iter, model, device=gpu_id))
    trainer.extend(training.extensions.PrintReport(
        ['epoch', 'main/loss',  'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(training.extensions.ProgressBar(update_interval=10))
    trainer.extend(training.extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(training.extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(training.extensions.dump_graph('main/loss'))

    trainer.run()

    chainer.serializers.save_npz("mymodel.npz", model)
    test_iter = iterators.SerialIterator(
        valid, batch_size, repeat=False, shuffle=False)
    #show_test_performance(model, test_iter, gpu_id,batch_size)

    result = {'y_pred': [],
              'y_true': []}
    for batch in test_iter:
        X_test, y_test = concat_examples(batch, gpu_id)
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            y_pred_batch = model.predictor(X_test)
        if gpu_id >= 0:
            y_pred_batch = chainer.cuda.to_cpu(y_pred_batch.data)
        result['y_pred'].extend(np.argmax(y_pred_batch, axis=1).tolist())
        result['y_true'].extend(y_test.tolist())
    print(confusion_matrix(result['y_true'], result['y_pred']))


if __name__ == '__main__':
    main()
