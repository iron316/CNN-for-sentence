#from datasets import create_dataset
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from chainer import iterators, training, optimizers
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


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
            self.fc = L.Linear(None, 2)

    def __call__(self, x):
        sentence_vec = x.reshape((-1, 1, 100, 200))
        h_3 = F.max_pooling_2d(F.tanh(self.cnn_w3(sentence_vec)), 100)
        h_4 = F.max_pooling_2d(F.tanh(self.cnn_w4(sentence_vec)), 100)
        h_5 = F.max_pooling_2d(F.tanh(self.cnn_w5(sentence_vec)), 100)
        concat = F.concat([h_3, h_4, h_5], axis=2)
        h2 = F.dropout(F.relu(concat), ratio=0.5)
        y = self.fc(h2)
        return y


def main():
    reset_seed(0)
    #dirs = ['natsume', 'edogawa']
    #data = create_dataset(dirs)
    #test, train_val = split_dataset_random(data, 10, seed=19910927)
    with open('sentence_vec.pickle', 'rb') as rbf:
        train = pickle.load(rbf)
    with open('test_sentence_vec.pickle', 'rb') as rbf:
        valid = pickle.load(rbf)
    #train, valid = split_dataset_random(data, int(len(data) * 0.8), seed=19910927)
    
    batch_size = 256
    gpu_id = 0
    max_epoch = 200

    train_iter = iterators.SerialIterator(train, batch_size)
    valid_iter = iterators.SerialIterator(
        valid, batch_size, repeat=False, shuffle=False)
    #test_iter = iterators.SerialIterator(
    #    test, batch_size, repeat=False, shuffle=False)

    net = CNN_fsc()

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
            'main/accuracy','validation/main/accuracy', 'elapsed_time']))
    trainer.extend(training.extensions.ProgressBar(update_interval=10))
    trainer.extend(training.extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(training.extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(training.extensions.dump_graph('main/loss'))

    trainer.run()

    show_test_performance(model, valid_iter, gpu_id,batch_size)
    #show_test_performance(model, test_iter, gpu_id,batch_size)


if __name__ == '__main__':
    main()
