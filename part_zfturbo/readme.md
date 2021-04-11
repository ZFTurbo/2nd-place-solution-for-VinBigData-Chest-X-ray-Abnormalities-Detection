# How to run

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
