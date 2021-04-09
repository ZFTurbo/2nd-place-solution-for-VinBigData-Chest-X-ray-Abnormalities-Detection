python3 ./net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold0.py
python3 ./net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold1.py
python3 ./net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold2.py
python3 ./net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold3.py
python3 ./net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold4.py

python3 ./net_v17_retinanet_finetune/r21_train_backbone_resnet101_sqr_fold0.py
python3 ./net_v17_retinanet_finetune/r21_train_backbone_resnet101_sqr_fold1.py
python3 ./net_v17_retinanet_finetune/r21_train_backbone_resnet101_sqr_fold2.py
python3 ./net_v17_retinanet_finetune/r21_train_backbone_resnet101_sqr_fold3.py
python3 ./net_v17_retinanet_finetune/r21_train_backbone_resnet101_sqr_fold4.py

python3 net_v09_yolo5/train.py --img 640 --batch 40 --epochs 80 --data ../../modified_data_folder/yolo5_data/fold_0.xml --device 0 --project ../../models_inference/yolov5_fold0/ --weights ../../models_inference/yolov5x.pt
python3 net_v09_yolo5/train.py --img 640 --batch 40 --epochs 80 --data ../../modified_data_folder/yolo5_data/fold_1.xml --device 0 --project ../../models_inference/yolov5_fold1/ --weights ../../models_inference/yolov5x.pt
python3 net_v09_yolo5/train.py --img 640 --batch 40 --epochs 80 --data ../../modified_data_folder/yolo5_data/fold_2.xml --device 0 --project ../../models_inference/yolov5_fold2/ --weights ../../models_inference/yolov5x.pt
python3 net_v09_yolo5/train.py --img 640 --batch 40 --epochs 80 --data ../../modified_data_folder/yolo5_data/fold_3.xml --device 0 --project ../../models_inference/yolov5_fold3/ --weights ../../models_inference/yolov5x.pt
python3 net_v09_yolo5/train.py --img 640 --batch 40 --epochs 80 --data ../../modified_data_folder/yolo5_data/fold_4.xml --device 0 --project ../../models_inference/yolov5_fold4/ --weights ../../models_inference/yolov5x.pt
