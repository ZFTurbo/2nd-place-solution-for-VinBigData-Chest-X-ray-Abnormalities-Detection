#!/usr/bin/env bash

./download_train_data.sh
python3 data/create_yolo_labels_separate_rads.py
python3 data/yolo_converter_stage2.py
python3 yolo_stage2_train.py --weights ./weights/stage1_fold0.pt --cfg ./yolo5/configYolo5/yolov5x_roma.yaml --data ./yolo5/configYolo5/vinbigdata0_stage2.yaml --hyp stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./runs/train --workers 2
python3 yolo_stage2_train.py --weights ./weights/stage1_fold1.pt --cfg ./yolo5/configYolo5/yolov5x_roma.yaml --data ./yolo5/configYolo5/vinbigdata1_stage2.yaml --hyp stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./runs/train --workers 2
python3 yolo_stage2_train.py --weights ./weights/stage1_fold2.pt --cfg ./yolo5/configYolo5/yolov5x_roma.yaml --data ./yolo5/configYolo5/vinbigdata2_stage2.yaml --hyp stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./runs/train --workers 2
python3 yolo_stage2_train.py --weights ./weights/stage1_fold3.pt --cfg ./yolo5/configYolo5/yolov5x_roma.yaml --data ./yolo5/configYolo5/vinbigdata3_stage2.yaml --hyp stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./runs/train --workers 2
python3 yolo_stage2_train.py --weights ./weights/stage1_fold4.pt --cfg ./yolo5/configYolo5/yolov5x_roma.yaml --data ./yolo5/configYolo5/vinbigdata4_stage2.yaml --hyp stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./runs/train --workers 2
cp ./runs/train/exp/weights/best.pt ./weights/stage2_fold0.pt
cp ./runs/train/exp2/weights/best.pt ./weights/stage2_fold1.pt
cp ./runs/train/exp3/weights/best.pt ./weights/stage2_fold2.pt
cp ./runs/train/exp4/weights/best.pt ./weights/stage2_fold3.pt
cp ./runs/train/exp5/weights/best.pt ./weights/stage2_fold4.pt
