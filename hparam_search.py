from ast import parse
import ray
from ray.tune.sample import loguniform, quniform
import tensorflow as tf
import numpy as np
import argparse
import warnings
import os

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.suggest.hebo import HEBOSearch

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import TriMelspecModel
from glob import glob

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


def train_sc(config):
    n_classes = len(os.listdir(config['src_root']))
   
    model = TriMelspecModel(
        n_classes=n_classes,
        sr=config['sr'],
        dt=config['dt'],
        backbone=config['backbone'],
        n_mels=config['n_mels'],
        spectrogram_width=config['spectrogram_width'],
        n_fft=config['n_fft'],
        dropout_1=config['dropout_1'],
        dropout_2=config['dropout_2'],
        dropout_3=config['dropout_3'],
        dropout_4=config['dropout_4'],
        dense_1=config['dense_1'],
        dense_2=config['dense_2'],
        dense_3=config['dense_3'],
        l2_lambda=config['l2_lambda'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        mask_pct=config['mask_pct'],
        mask_thresh=config['mask_thresh'],
        activation=config['activation']
    )

    wav_paths = glob(f'{config["src_root"]}/**', recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

    classes = sorted(os.listdir(config['src_root']))
    le = LabelEncoder()
    le.fit(classes)

    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    if config['val_root']:
        wav_val = glob(f'{config["val_root"]}/**', recursive=True)
        wav_val = [x.replace(os.sep, '/') for x in wav_val if '.wav' in x]

        label_val = [os.path.split(x)[0].split('/')[-1] for x in wav_val]
        label_val = le.transform(label_val)

        wav_train, label_train = wav_paths, labels
    else:
        wav_train, wav_val, label_train, label_val = train_test_split(wav_paths, labels, test_size=0.1, random_state=0)

    assert len(label_train) >= config['batch_size'], 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != n_classes:
        warnings.warn(f"Found {len(set(label_train))}/{n_classes} classes in training data. Increase data size or change random_state.")
    if len(set(label_val)) != n_classes:
        warnings.warn(f"Found {len(set(label_val))}/{n_classes} classes in validation data. Increase data size or change random_state.")

    tg = DataGenerator(wav_train, label_train, config['sr'], config['dt'], n_classes, batch_size=config['batch_size'])
    vg = DataGenerator(wav_val, label_val, config['sr'], config['dt'], n_classes, batch_size=config['batch_size'])
    cp_best_val_acc = ModelCheckpoint(os.path.join(ray.tune.get_trial_dir(), 'best_val_acc.h5'), monitor='val_accuracy',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    cp_best_val_loss = ModelCheckpoint(os.path.join(ray.tune.get_trial_dir(), 'best_val_loss.h5'), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, verbose=1)

    print(ray.tune.get_trial_dir())

    model.fit(tg, validation_data=vg,
              epochs=config['epochs'], verbose=1,
              callbacks=[reduce_lr, TuneReportCallback({
                  'accuracy': 'accuracy',
                  'val_accuracy': 'val_accuracy'
              }), cp_best_val_acc, cp_best_val_loss])
    
def tune_sc(args):
    analysis = tune.run(
        train_sc,
        name=args.expt_name,
        scheduler=AsyncHyperBandScheduler(time_attr='training_iteration', metric='val_accuracy', mode='max', max_t=40, grace_period=15),
        search_alg=HEBOSearch(metric='val_accuracy', mode='max', max_concurrent=1),
        stop={
            'training_iteration': args.num_training_iterations
        },
        num_samples=args.num_samples,
        resources_per_trial={
            'cpu': 2,
            'gpu': 1
        },
        keep_checkpoints_num=1,
        checkpoint_score_attr='val_accuracy',
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=args.output_root,
        config={
            'src_root': args.src_root,
            'val_root': args.val_root,
            'dt': 1.0,
            'sr': 22050,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'spectrogram_width': tune.choice([250, 375, 500]),
            'n_mels': tune.choice([128, 256]),
            'n_fft': tune.choice([2048, 4096]),
            'dropout_1': tune.quniform(0.1, 0.4, 0.1),
            'dropout_2': tune.quniform(0.1, 0.4, 0.1),
            'dropout_3': tune.quniform(0.1, 0.5, 0.1),
            'dropout_4': tune.quniform(0.1, 0.5, 0.1),
            'dense_1': tune.choice([512, 1024, 2048]),
            'dense_2': tune.choice([512, 1024, 2048]),
            'dense_3': tune.choice([512, 1024, 2048]),
            'l2_lambda': tune.loguniform(1e-6, 1e-3),
            'mask_pct': tune.quniform(0.1, 0.4, 0.1),
            'mask_thresh': tune.uniform(0.1, 0.7),
            'learning_rate': tune.uniform(1e-4, 5e-3),
            'activation': tune.choice(['mish', 'tanh', 'relu', 'sigmoid']),
            'backbone': tune.choice(['densenet169', 'densenet121', 'resnet152', 'efficientnetb7', 'efficientnetv2-l'])
        }
    )
    print('Best hyperparameters found were: ', analysis.best_config)

if __name__ == '__main__':
    ray.init(num_gpus=1)
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--src_root', type=str, default='clean', help='directory of audio files in total duration')
    parser.add_argument('--val_root', type=str, default=None, help='directory of audio files used for val')
    parser.add_argument('--output_root', type=str, default='runs', help='directory to store output model files and logs')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to do')
    parser.add_argument('--batch_size', type=int, default=26, help='batch size')
    parser.add_argument('--num_training_iterations', type=int)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--restore_root', type=str)
    parser.add_argument('--expt_name', type=str)
    args, _ = parser.parse_known_args()
    tune_sc(args)