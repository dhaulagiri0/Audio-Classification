import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import warnings
import os
import datetime
import pdb

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import ConvDense
from glob import glob
from tensorboard.plugins.hparams import api as hp

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    hparams = {
        hp.HParam('n_mels'): args.n_mels,
        hp.HParam('spectrogram_width'): args.spectrogram_width,
        hp.HParam('spectrogram_height'): args.spectrogram_height,
        hp.HParam('n_fft'): args.n_fft,
        hp.HParam('dropout_1'): args.dropout_1,
        hp.HParam('dropout_2'): args.dropout_2,
        hp.HParam('n_neurons'): args.n_neurons,
        hp.HParam('l2_lambda'): args.l2_lambda
    }
    n_classes = len(os.listdir(args.src_root))
    model = ConvDense(n_classes, args.sr, args.dt, args.n_mels, args.spectrogram_width, args.spectrogram_height, args.n_fft, args.dropout_1, args.dropout_2, args.n_neurons, args.l2_lambda, args.batch_size)

    wav_paths = glob(f'{args.src_root}/**', recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths, labels, test_size=0.1, random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != n_classes:
        warnings.warn(f"Found {len(set(label_train))}/{n_classes} classes in training data. Increase data size or change random_state.")
    if len(set(label_val)) != n_classes:
        warnings.warn(f"Found {len(set(label_val))}/{n_classes} classes in validation data. Increase data size or change random_state.")

    tg = DataGenerator(wav_train, label_train, args.sr, args.dt, n_classes, batch_size=args.batch_size)
    vg = DataGenerator(wav_val, label_val, args.sr, args.dt, n_classes, batch_size=args.batch_size)
    runtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cp_best_val_acc = ModelCheckpoint(os.path.join(args.output_root, runtime, 'best_val_acc.h5'), monitor='val_accuracy',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    cp_best_val_loss = ModelCheckpoint(os.path.join(args.output_root, runtime, 'best_val_loss.h5'), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    tb = TensorBoard(os.path.join(args.output_root, runtime, 'logs'), histogram_freq=1)
    hparams_dir = os.path.join(args.output_root, runtime, 'logs', 'validation')
    with tf.summary.create_file_writer(hparams_dir).as_default():
        hp.hparams_config(
            hparams=hparams,
            metrics=[hp.Metric('epoch_accuracy')]
        )
    hparams_cb = hp.KerasCallback(writer=hparams_dir, hparams=hparams)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)

    model.fit(tg, validation_data=vg,
              epochs=args.epochs, verbose=1,
              callbacks=[cp_best_val_acc, cp_best_val_loss, tb, hparams_cb, reduce_lr, early_stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--src_root', type=str, default='clean', help='directory of audio files in total duration')
    parser.add_argument('--output_root', type=str, default='runs', help='directory to store output model files and logs')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to do')
    parser.add_argument('--dt', type=float, default=1.0, help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=22050, help='sample rate of clean audio')
    # hyperparameters to try
    parser.add_argument('--batch_size', type=int, default=15, help='batch size')
    parser.add_argument('--n_mels', type=int, default=128, help='number of melspectrograms')
    parser.add_argument('--spectrogram_width', type=int, default=250, help='width of resized melspectrogram')
    parser.add_argument('--spectrogram_height', type=int, default=128, help='height of resized melspectrogram')
    parser.add_argument('--n_fft', type=int, help='number of fast fourier transform frequencies to analyze')
    parser.add_argument('--dropout_1', type=float, help='dropout rate between densenet and FCL')
    parser.add_argument('--dropout_2', type=float, help='dropout rate between FCL and last layer')
    parser.add_argument('--n_neurons', type=int, help='number of neurons in fully connected layer')    
    parser.add_argument('--l2_lambda', type=float, help='l2 regularization lambda')

    args, _ = parser.parse_known_args()
    train(args)

