#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python predict.py --model_fn=ray_results/sc_expt_2dense_full/train_sc_fc9ad3b0_13_activation=relu,backbone=densenet121,batch_size=26,dense_1=1024,dense_2=1024,dropout_1=0.19375,dropout_2=0.15_2021-06-15_16-45-07/best_val_acc.h5 --src_dir=../s1_test_cleaned --dt=1.0 --sr=22050