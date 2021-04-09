#!/usr/bin/env bash

for i in {0..4}
do
  python mmdetection/tools/train.py configs/train/cascade_rcnn_r50_rfp_1x_xray_stage1_fold$i.py
done

for i in {0..4}
do
  python mmdetection/tools/train.py configs/train/cascade_rcnn_r50_rfp_1x_xray_stage2_fold$i.py
done
