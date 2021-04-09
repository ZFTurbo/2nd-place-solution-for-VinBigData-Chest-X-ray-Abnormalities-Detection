#!/usr/bin/env bash

pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection
pip install -r requirements/build.txt
pip install -r requirements/optional.txt
pip install future tensorboard
pip install -v -e .
cd ..
cp schedule_1x.py mmdetection/configs/_base_/schedules/