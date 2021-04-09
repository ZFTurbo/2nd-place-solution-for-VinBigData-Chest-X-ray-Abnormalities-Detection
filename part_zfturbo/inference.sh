python3 ./net_v04_retinanet/r60_get_vectors_resnet101_sqr.py
python3 ./net_v04_retinanet/r61_create_csvs_resnet101_sqr.py
python3 ./net_v17_retinanet_finetune/r60_get_vectors_resnet101_sqr.py
python3 ./net_v17_retinanet_finetune/r61_create_csvs_resnet101_sqr.py

python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold0/ --project ../../modified_data_folder/yolov5_fold0/ --name valid_iou_0.25_0.01
python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold0/ --name test_iou_0.25_0.01

python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_1_mAP_0.395_0.182.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold1/ --project ../../modified_data_folder/yolov5_fold1/ --name valid_iou_0.25_0.01
python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_1_mAP_0.395_0.182.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold1/ --name test_iou_0.25_0.01

python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_2_mAP_0.415_0.196.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold2/ --project ../../modified_data_folder/yolov5_fold2/ --name valid_iou_0.25_0.01
python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_2_mAP_0.415_0.196.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold2/ --name test_iou_0.25_0.01

python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_3_mAP_0.382_0.183.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold3/ --project ../../modified_data_folder/yolov5_fold3/ --name valid_iou_0.25_0.01
python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_3_mAP_0.382_0.183.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold3/ --name test_iou_0.25_0.01

python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_4_mAP_0.409_0.189.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold4/ --project ../../modified_data_folder/yolov5_fold4/ --name valid_iou_0.25_0.01
python3 net_v09_yolo5/detect.py --weights ../../models_inference/yolo_best/best_fold_4_mAP_0.409_0.189.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold4/ --name test_iou_0.25_0.01

python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold0/ --project ../../modified_data_folder/yolov5_fold0/ --name valid_iou_0.25_0.01_mirror
python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_0_mAP_0.383_0.184.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold0/ --name test_iou_0.25_0.01_mirror

python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_1_mAP_0.395_0.182.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold1/ --project ../../modified_data_folder/yolov5_fold1/ --name valid_iou_0.25_0.01_mirror
python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_1_mAP_0.395_0.182.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold1/ --name test_iou_0.25_0.01_mirror

python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_2_mAP_0.415_0.196.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold2/ --project ../../modified_data_folder/yolov5_fold2/ --name valid_iou_0.25_0.01_mirror
python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_2_mAP_0.415_0.196.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold2/ --name test_iou_0.25_0.01_mirror

python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_3_mAP_0.382_0.183.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold3/ --project ../../modified_data_folder/yolov5_fold3/ --name valid_iou_0.25_0.01_mirror
python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_3_mAP_0.382_0.183.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold3/ --name test_iou_0.25_0.01_mirror

python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_4_mAP_0.409_0.189.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/train_fold4/ --project ../../modified_data_folder/yolov5_fold4/ --name valid_iou_0.25_0.01_mirror
python3 net_v09_yolo5/detect_mirror.py --weights ../../models_inference/yolo_best/best_fold_4_mAP_0.409_0.189.pt --device 1 --img-size 640 --save-txt --save-conf --conf-thres 0.01 --iou-thres 0.25 --source ../../input/yolo/images/test/ --project ../../modified_data_folder/yolov5_fold4/ --name test_iou_0.25_0.01_mirror

python3 net_v09_yolo5/convert_yolo_preds.py
python3 r16_ensemble_predictions.py