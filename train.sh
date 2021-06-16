#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python train.py \
    --src_root="/home/cheongalc/Documents/til2021/train_cleaned" \
    --val_root="/home/cheongalc/Documents/til2021/test_cleaned_sorted" \
    --output_root="/home/cheongalc/Documents/til2021/Audio-Classification/runs/" \
    --epochs=300 \
    --dt=1.0 \
    --sr=22050 \
    --backbone="resnet152" \
    --batch_size=26 \
    --n_mels=128 \
    --spectrogram_width=250 \
    --n_fft=2048 \
    --dropout_1=0.28750 \
    --dropout_2=0.13750 \
    --dropout_3=0.45 \
    --dense_1=512 \
    --dense_2=1024 \
    --l2_lambda=0.000074989 \
    --mask_pct=0.28750 \
    --mask_thresh=0.62500 \
    --learning_rate=0.00071250 \
    --activation="mish"