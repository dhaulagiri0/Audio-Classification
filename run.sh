#!/bin/bash
source $HOME/venvs/til2021/bin/activate

for spectrogram_width in 375 500
do
    for n_fft in 2048 4096
    do
        for dropout_1 in 0.2 0.3 0.4
        do 
            for dropout_2 in 0.2 0.3 0.4
            do
                for n_neurons in 1024 2048
                do
                    for l2_lambda in 0.01 0.001
                    do
                        python train.py \
                        --src_root='../train' \
                        --output_root='runs' \
                        --epochs=1 \
                        --spectrogram_width=$spectrogram_width \
                        --n_fft=$n_fft \
                        --n_dropout_1=$n_dropout_1 \
                        --n_dropout_2=$n_dropout_2 \
                        --n_neurons=$n_neurons \
                        --l2_lambda=$l2_lambda
                    done
                done
            done
        done
    done
done
                