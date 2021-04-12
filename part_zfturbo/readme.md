# How to run

First you need to put [competition data](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data) in `../input/` folder. It will require around 190 GB of space. Preprocessing and storage of models could require plus 200 GB of space. 

`pip install -r requirements.txt`

## Inference 

```
sh ./preproc.sh
sh ./inference.sh
```

## Train

```
sh ./preproc.sh
sh ./train.sh
sh ./inference.sh
```

## Result
As the result of code there will be generated 3 CSV files with predictions of 3 models on contest test data:
* `../subm_folder/ensemble_retinanet_resnet101_sqr.csv`
* `../subm_folder/ensemble_retinanet_resnet101_removed_rad.csv`
* `../subm_folder/ensemble_yolo_final.csv`
