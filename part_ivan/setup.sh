#!/usr/bin/env bash

pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && git checkout 375605fba8f89f40eb1b6b67b4aab83fbe769098
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection && git checkout 5ebba9a9221c3f6f8754d4efe293cdcd12be8dd4
cp ../build.txt requirements/
cp ../optional.txt requirements/
pip install -r requirements/build.txt
pip install -r requirements/optional.txt
pip install future==0.18.2 tensorboard==2.4.1
pip install -v -e .
cd ..
cp schedule_1x.py mmdetection/configs/_base_/schedules/
