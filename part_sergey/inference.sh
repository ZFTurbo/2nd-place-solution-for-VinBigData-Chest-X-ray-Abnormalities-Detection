#!/usr/bin/env bash

./download_test_data.sh
python3 yolo_inf.py --stage 2
cp ./yolo_stage2_all_folds.csv ../subm_folder/yolo_stage2_all_folds.csv
