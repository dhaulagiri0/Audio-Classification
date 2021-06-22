import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import warnings
import os
import datetime
import pdb

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import TriMelspecModel, EnsembleModel, TriSpecModel, WavegramCNN, mish, ChangeModelHead
from augmentation_layers import RandomFreqMask, RandomTimeMask, RandomNoise, RandomTimeShift
from glob import glob
import tensorflow_hub as hub
import librosa
import numpy as np

def pitch_shift_numpy(x, sampling_rate, n_steps=3):
    x = np.array(x)
    x = np.squeeze(x)
    curr_n_steps = int(tf.random.uniform(shape=(), minval=0, maxval=1) * n_steps) 
    return librosa.effects.pitch_shift(x, sampling_rate, curr_n_steps)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True, percentage=0.8):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.percentage=percentage
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
            rate, wave = wavfile.read(path)
            wave = wave.astype('float32')
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

            c = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float16)
            if c <= self.percentage:
                X[i,] = pitch_shift_numpy(wave, sampling_rate=self.sr).reshape(-1, 1)
            else:
                X[i,] = wave.reshape(-1, 1)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    n_classes = len(os.listdir(args.src_root))
  
    if args.ensemble:
        model = EnsembleModel(
            model_paths=args.ensemble_paths,
            n_classes=n_classes,            
            sr=args.sr,
            dt=args.dt,           
            l2_lambda=args.l2_lambda,
            learning_rate=args.learning_rate,
            dropout_1=args.dropout_1,
            dropout_2=args.dropout_2,
            dropout_3=args.dropout_3,
            dropout_4=args.dropout_4,
            connector_dense=1024,
            dense_1=args.dense_1,
            dense_2=args.dense_2,
            dense_3=args.dense_3,
            activation=args.activation) 
        if args.weights:
            model.load_weights(args.weights)
    if args.weights:
        KerasLayer = hub.KerasLayer('gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/feature-vector', trainable=True)
        model = load_model(args.weights,
                custom_objects={'STFT':STFT,
                                'Magnitude':Magnitude,
                                'ApplyFilterbank':ApplyFilterbank,
                                'MagnitudeToDecibel':MagnitudeToDecibel,
                                'RandomTimeMask': RandomTimeMask,
                                'RandomFreqMask': RandomFreqMask,
                                'mish':mish,
                                'KerasLayer': KerasLayer,
                                'RandomNoise':RandomNoise,
                                'RandomTimeShift': RandomTimeShift})
        if args.new_n_classes:
            model = ChangeModelHead(
                model, 
                args.new_n_classes, 
                args.learning_rate,
                dropout_1=args.dropout_1,
                dropout_2=args.dropout_2,
                dropout_3=args.dropout_3,
                dropout_4=args.dropout_4,
                dense_1=args.dense_1,
                dense_2=args.dense_2,
                dense_3=args.dense_3,
                l2_lambda=args.l2_lambda, 
                activation=args.activation)
    else:
        if args.model == 'trimelspec':
            model = TriMelspecModel(
                n_classes=n_classes,
                sr=args.sr,
                dt=args.dt,
                backbone=args.backbone,
                n_mels=args.n_mels,
                spectrogram_width=args.spectrogram_width,
                n_fft=args.n_fft,
                dropout_1=args.dropout_1,
                dropout_2=args.dropout_2,
                dropout_3=args.dropout_3,
                dropout_4=args.dropout_4,
                dense_1=args.dense_1,
                dense_2=args.dense_2,
                dense_3=args.dense_3,
                l2_lambda=args.l2_lambda,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                mask_pct=args.mask_pct,
                mask_thresh=args.mask_thresh,
                activation=args.activation
            )
        elif args.model == 'trispec':
            model = TriSpecModel(
                n_classes=n_classes,
                sr=args.sr,
                dt=args.dt,
                backbone=args.backbone,
                spectrogram_width=args.spectrogram_width,
                spectrogram_height=args.spectrogram_height,
                n_fft=args.n_fft,
                dropout_1=args.dropout_1,
                dropout_2=args.dropout_2,
                dropout_3=args.dropout_3,
                dropout_4=args.dropout_4,
                dense_1=args.dense_1,
                dense_2=args.dense_2,
                dense_3=args.dense_3,
                l2_lambda=args.l2_lambda,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                mask_pct=args.mask_pct,
                mask_thresh=args.mask_thresh,
                activation=args.activation,
                return_decibel=args.return_decibel
            )
        else:
            model = WavegramCNN(
                n_classes=n_classes,
                sr=args.sr,
                dt=args.dt,
                backbone=args.backbone,
                n_mels=args.n_mels,
                spectrogram_width=args.spectrogram_width,
                n_fft=args.n_fft,
                dropout_1=args.dropout_1,
                dropout_2=args.dropout_2,
                dropout_3=args.dropout_3,
                # dropout_4=args['dropout_4'],
                dense_1=args.dense_1,
                dense_2=args.dense_2,
                # dense_3=args['dense_3'],
                l2_lambda=args.l2_lambda,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                mask_pct=args.mask_pct,
                mask_thresh=args.mask_thresh,
                activation=args.activation
            )
      

    wav_paths = glob(f'{args.src_root}/**', recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)

    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    if args.val_root:
        wav_val = glob(f'{args.val_root}/**', recursive=True)
        wav_val = [x.replace(os.sep, '/') for x in wav_val if '.wav' in x]

        label_val = [os.path.split(x)[0].split('/')[-1] for x in wav_val]
        label_val = le.transform(label_val)

        wav_train, label_train = wav_paths, labels

    else:
        wav_train, wav_val, label_train, label_val = train_test_split(wav_paths, labels, test_size=0.1, random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != n_classes:
        warnings.warn(f"Found {len(set(label_train))}/{n_classes} classes in training data. Increase data size or change random_state.")
    if len(set(label_val)) != n_classes:
        warnings.warn(f"Found {len(set(label_val))}/{n_classes} classes in validation data. Increase data size or change random_state.")

    tg = DataGenerator(wav_train, label_train, args.sr, args.dt, n_classes, batch_size=args.batch_size, percentage=0.8)
    vg = DataGenerator(wav_val, label_val, args.sr, args.dt, n_classes, batch_size=args.validation_batch_size, percentage=0.8)
    runtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + args.run_name
    cp_best_val_acc = ModelCheckpoint(os.path.join(args.output_root, runtime, 'best_val_acc.h5'), monitor='val_accuracy',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    cp_best_val_loss = ModelCheckpoint(os.path.join(args.output_root, runtime, 'best_val_loss.h5'), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    tb = TensorBoard(os.path.join(args.output_root, runtime, 'logs'), histogram_freq=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)

    model.fit(tg, validation_data=vg,
            epochs=args.epochs, verbose=1,
            callbacks=[cp_best_val_acc, cp_best_val_loss, tb, reduce_lr, early_stopping], validation_batch_size=args.validation_batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--src_root', type=str, default='s1_train', help='directory of audio files used for train')
    parser.add_argument('--val_root', type=str, default='s1_val', help='directory of audio files used for val')
    parser.add_argument('--output_root', type=str, default='runs', help='directory to store output model files and logs')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to do')
    parser.add_argument('--dt', type=float, default=1.0, help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=22050, help='sample rate of clean audio')
    parser.add_argument('--run_name', type=str, default='') 
    parser.add_argument('--model', type=str, default='trimelspec')
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--batch_size', type=int, default=26, help='batch size')
    parser.add_argument('--validation_batch_size', type=int, default=15)
    parser.add_argument('--n_mels', type=int, default=128, help='number of melspec bins')
    parser.add_argument('--spectrogram_width', type=int, default=250, help='width of resized spectrogram')
    parser.add_argument('--spectrogram_height', type=int, default=512)
    parser.add_argument('--n_fft', type=int, default=2048, help='number of fast fourier transform frequencies to analyze')
    parser.add_argument('--dropout_1', type=float, default=0.2)
    parser.add_argument('--dropout_2', type=float, default=0.2)
    parser.add_argument('--dropout_3', type=float, default=0.0)
    parser.add_argument('--dropout_4', type=float, default=0.0)
    parser.add_argument('--dense_1', type=int, default=1024, help='number of neurons in fully connected layer')  
    parser.add_argument('--dense_2', type=int, default=0)
    parser.add_argument('--dense_3', type=int, default=0)
    parser.add_argument('--l2_lambda', type=float, default=0.001, help='l2 regularization lambda')
    parser.add_argument('--mask_pct', type=float, default=0.2)
    parser.add_argument('--mask_thresh', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--weights', default=None, help='path of the model weights to resume from', type=str)
    parser.add_argument('--ensemble', default=False, action='store_true')
    parser.add_argument('--ensemble_paths', action='append')
    parser.add_argument('--return_decibel', type=bool, default=True)
    parser.add_argument('--new_n_classes', default=None, type=int, help='number of classes to predict')
    args, _ = parser.parse_known_args()
    train(args)