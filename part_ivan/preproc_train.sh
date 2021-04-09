#!/usr/bin/env bash

wget http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth
mkdir checkpoints && mv cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth checkpoints/

kaggle datasets download ivanpan/vinbigdata-train-data
unzip vinbigdata-train-data.zip -d data
