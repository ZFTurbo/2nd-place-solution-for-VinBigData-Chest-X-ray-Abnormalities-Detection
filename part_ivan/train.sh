#!/usr/bin/env bash

for i in {0..4}
do
  python mmdetection/tools/train.py configs/train/cascade_rcnn_r50_rfp_1x_xray_stage1_fold$i.py
done

for i in {0..4}
do
  python mmdetection/tools/train.py configs/train/cascade_rcnn_r50_rfp_1x_xray_stage2_fold$i.py
done

mkdir cascade_r50_augs_with_empty cascade_r50_augs_rare_with_empty
for i in {0..4}
do
  cp work_dirs/cascade_r50_augs_with_empty_fold$i/best_bbox_mAP_50.pth cascade_r50_augs_with_empty/fold$i.pth
  cp work_dirs/cascade_r50_augs_rare_with_empty_fold$i/best_bbox_mAP_50.pth cascade_r50_augs_rare_with_empty/fold$i.pth
done
