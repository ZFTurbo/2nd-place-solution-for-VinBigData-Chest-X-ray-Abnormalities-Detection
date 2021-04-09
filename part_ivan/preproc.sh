#!/usr/bin/env bash

kaggle datasets download ivanpan/vinbigdata-weights
unzip vinbigdata-weights.zip -d weights/

kaggle datasets download ivanpan/vinbigdata-data
unzip vinbigdata-data.zip
