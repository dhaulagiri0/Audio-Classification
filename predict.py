
import tensorflow as tf
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from augmentation_layers import RandomFreqMask, RandomTimeMask, RandomNoise, RandomTimeShift
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
from models import TriMelspecModel, mish, EnsembleModel
import tensorflow_hub as hub

# custom predict function to predict test data
def predict_test(args):
    # load model
    if args.ensemble:
      model = EnsembleModel(
              model_paths=args.ensemble_paths,
              n_classes=13,            
              sr=args.sr,
              dt=args.dt,           
              l2_lambda=0.0001,
              learning_rate=1e-4,)
      model.load_weights(args.model_fn)
    else:
      KerasLayer = hub.KerasLayer('gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/feature-vector', trainable=True)
      model = load_model(args.model_fn,
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

    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    preds = []
    file_names = []
    pred_mean = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)

        # class list for s1
        # classes = ['bird', 'eight', 'falcon', 'five', 'four', 'nine', 'one', 'seven', 'six', 'snake', 'three', 'two', 'zero']
        
        # class list for s2
        classes = ['bird', 'cat', 'chicken', 'dog', 'down', 'eight', 'falcon', 'five', 'four', 'go', 'left', 'nine', 'one', 'right', 'seven', 'six', 'snake', 'stop', 'three', 'two', 'up', 'zero']
        
        pred = classes[y_pred]
        file_name = wav_paths[z].split('/')[-1]
        print('File: {} Predicted class: {}'.format(file_name, pred))
        preds.append(pred)
        file_names.append(file_name)
        pred_mean.append(y_mean)

    df = pd.DataFrame({'a':file_names, 'b':preds})
    df.to_csv(os.path.join('preds', args.pred_fn + '.csv'), index=False, header=False)

    df = pd.DataFrame({'a':file_names, 'b':pred_mean})
    df.to_csv(os.path.join('preds', args.pred_fn + '_mean.csv'), index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Prediction')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5', help='model filename to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred', help='filename to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles', help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0, help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000, help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20, help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--efficientnet', type=bool)
    parser.add_argument('--ensemble', default=False, action='store_true')
    parser.add_argument('--ensemble_paths', action='append')
    args, _ = parser.parse_known_args()

    predict_test(args)

