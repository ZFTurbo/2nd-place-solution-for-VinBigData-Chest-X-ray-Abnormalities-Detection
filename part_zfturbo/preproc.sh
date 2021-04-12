python3 ./preproc_data/r01_extract_meta.py
python3 ./preproc_data/r02_create_kfold_split.py
python3 ./preproc_data/r03_convert_dicom_to_png.py
python3 ./preproc_data/r05_merge_boxes_v1.py
python3 ./preproc_data/r08_prepare_yolo_dataset.py
python3 ./preproc_data/r09_squarify_boxes.py
python3 ./preproc_data/r12_extract_selected_radiologists.py
mkdir ../models_inference/
wget https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/releases/download/v1.0/yolov5x.pt -P ../models_inference/
wget https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/releases/download/v1.0/retinanet_resnet101_500_classes_0.4986.h5 -P ../models_inference/
wget https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/releases/download/v1.0/retinanet_resnet101_sqr.zip -P ../models_inference/
wget https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/releases/download/v1.0/retinanet_resnet101_sqr_removed_rads.zip -P ../models_inference/
wget https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection/releases/download/v1.0/yolo_best.zip -P ../models_inference/
unzip ../models_inference/yolo_best.zip -d  ../models_inference/yolo_best/
unzip ../models_inference/retinanet_resnet101_sqr.zip -d  ../models_inference/retinanet_resnet101_sqr/
unzip ../models_inference/retinanet_resnet101_sqr_removed_rads.zip -d  ../models_inference/retinanet_resnet101_sqr_removed_rads/
