from datasets import create_dataset
import chainer
from chainer import iterators, training, optimizers
import nets
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gensim.models import KeyedVectors
import argparse
import json
plt.switch_backend('agg')


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def get_w(model):
    w = model.vectors
    unk_vec = np.random.normal(size=(200)).astype(np.float32)
    w2v_w = np.vstack((w, unk_vec))
    return w2v_w


def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = chainer.cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                  for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = chainer.cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}
    else:
        return to_device_batch([x for x in batch])


class Preprocess(chainer.dataset.DatasetMixin):
    def __init__(self, values, ratio):
        self.values = values
        self.ratio = ratio

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        value, label = self.values[i]
        drop_value = [X if np.random.random() > self.ratio else -1 for X in value]
        return (np.array(drop_value, dtype=np.int32), label)


def main():
    parser = argparse.ArgumentParser(
        description='CNN for sentence classifier')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-model', default='TwoChannel',
                        choices=['NonStatic', 'Static', 'TwoChannel'],
                        help='Name of encoder model type.')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    reset_seed(0)
    model_dir = "entity_vector.model.bin"
    w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    train_dirs = ['natsume', 'edogawa', 'dazai', 'akutagawa', 'miyazawa']
    test_dirs = ['test_natsume', 'test_edogawa',
                 'test_dazai', 'test_akutagawa', 'test_miyazawa']
    train = create_dataset(train_dirs, w2v_model)
    valid = create_dataset(test_dirs, w2v_model)
    train = Preprocess(train, ratio=0.2)

    batch_size = args.batchsize
    gpu_id = args.gpu
    max_epoch = args.epoch

    w2v_w = get_w(w2v_model)

    train_iter = iterators.MultithreadIterator(train, batch_size, n_threads=4)
    valid_iter = iterators.MultithreadIterator(
        valid, batch_size, n_threads=4, repeat=False, shuffle=False)
    if args.model == 'Non_static':
        Encoder = nets.Non_static
    elif args.model == 'Static':
        Encoder = nets.Static
    elif args.model == 'Two_channel':
        Encoder = nets.Two_channel
    encoder = Encoder(w2v_w, batch_size)

    model = nets.TextClassifier(encoder)

    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    optimizer = optimizers.Adam().setup(model)

    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert_seq, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="result")

    trainer.extend(training.extensions.LogReport())
    trainer.extend(training.extensions.Evaluator(
        valid_iter, model, converter=convert_seq, device=gpu_id))
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

    result = {'y_pred': [],
              'y_true': []}
    for batch in test_iter:
        test = convert_seq(batch, gpu_id)
        X_test = test['xs']
        y_test = [int(y[0]) for y in test['ys']]
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            y_pred_batch = model.predict(X_test)
        if gpu_id >= 0:
            y_pred_batch = chainer.cuda.to_cpu(y_pred_batch.data)
        result['y_pred'].extend(np.argmax(y_pred_batch, axis=1).tolist())
        result['y_true'].extend(y_test)
    print(confusion_matrix(result['y_true'], result['y_pred']))


if __name__ == '__main__':
    main()
