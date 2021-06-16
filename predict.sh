#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python predict_efficient.py --model_fn=runs/colab_efficientnetv2l/best_val_acc.h5 --src_dir=../s1_test_cleaned --dt=1.0 --sr=22050