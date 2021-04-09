#!/usr/bin/env bash

mkdir -p pkl_preds/cascade_r50_augs_rare_with_empty/test pkl_preds/cascade_r50_augs_with_empty/test final_subs

for i in {0..4}
do
 python mmdetection/tools/test.py configs/test/cascade_rcnn_r50_rfp_1x_xray_test.py weights/cascade_r50_augs_rare_with_empty/fold$i.pth --out pkl_preds/cascade_r50_augs_rare_with_empty/test/fold$i.pkl
 python mmdetection/tools/test.py configs/test/cascade_rcnn_r50_rfp_1x_xray_test.py weights/cascade_r50_augs_with_empty/fold$i.pth --out pkl_preds/cascade_r50_augs_with_empty/test/fold$i.pkl
done

python data_scripts/pkl_to_subs.py --model-name cascade_r50_augs_rare_with_empty --resolution 1024
python data_scripts/pkl_to_subs.py --model-name cascade_r50_augs_with_empty --resolution 1024

python data_scripts/blend_subs.py --model-name cascade_r50_augs_rare_with_empty --resolution 1024
python data_scripts/blend_subs.py --model-name cascade_r50_augs_with_empty --resolution 1024

python data_scripts/postprocess_v2.py --model-name cascade_r50_augs_rare_with_empty --resolution 1024
python data_scripts/postprocess_v2.py --model-name cascade_r50_augs_with_empty --resolution 1024

cp subs/cascade_r50_augs_rare_with_empty_1024/cascade_r50_augs_rare_with_empty_5folds_1024_postprocess.csv final_subs/
cp subs/cascade_r50_augs_with_empty_1024/cascade_r50_augs_with_empty_5folds_1024_postprocess.csv final_subs/
