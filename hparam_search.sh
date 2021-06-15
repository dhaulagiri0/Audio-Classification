#!/bin/bash
source $HOME/venvs/til2021/bin/activate

python hparam_search.py --src_root=/home/cheongalc/Documents/til2021/train_cleaned --val_root=/home/cheongalc/Documents/til2021/test_cleaned_sorted --output_root=/home/cheongalc/Documents/til2021/Audio-Classification/ray_results/ --epochs=25 --batch_size=26 --num_training_iterations=25 --num_samples=30 