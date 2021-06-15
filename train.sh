#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python train.py --src_root=/home/cheongalc/Documents/til2021/train_cleaned --val_root=/home/cheongalc/Documents/til2021/test_cleaned_sorted --output_root=/home/cheongalc/Documents/til2021/Audio-Classification/runs/ --epochs=300 --dt=1.0 --sr=22050 --batch_size=26 --n_mels=128 --spectrogram_width=250 --n_fft=2048