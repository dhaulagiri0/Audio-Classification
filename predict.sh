#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python predict.py --model_fn=runs/20210615-120355/best_val_loss.h5 --src_dir=../s1_test_cleaned --dt=1.0 --sr=22050