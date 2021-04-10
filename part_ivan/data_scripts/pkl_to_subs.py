import pandas as pd
import numpy as np
import cv2
import os
import argparse
import json

label_ids_to_names = ['Aortic_enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
           'Lung_Opacity', 'Nodule/Mass',
           'Other_lesion', 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax', 'Pulmonary_fibrosis']


def upscale_bbox(bbox, width, height, im_size):
    x_scale = width / im_size
    y_scale = height / im_size

    bbox[0] *= x_scale
    bbox[2] *= x_scale
    bbox[1] *= y_scale
    bbox[3] *= y_scale

    return [int(x) for x in bbox]


def create_sub(preds, image_paths, original_test, resolution):
    sub_strings = []
    for pred, path in zip(preds, image_paths):
        width, height = original_test.loc[original_test.image_id == path.split('.')[0]].values[0][-2:]
        sub_string = ""
        for index, class_id in enumerate(pred):
            for detect in class_id:
                [x1, y1, x2, y2, c] = detect
                [x1, y1, x2, y2] = upscale_bbox([x1, y1, x2, y2], width, height, resolution)
                # x2, y2 = x1+w, y1+h
                if sub_string == "":
                    sub_string = '{} {} {} {} {} {}'.format(index, c, int(x1), int(y1), int(x2), int(y2))
                else:
                    sub_string += ' {} {} {} {} {} {}'.format(index, c, int(x1), int(y1), int(x2), int(y2))

        sub_strings.append(sub_string)

    sub = pd.DataFrame(
        {'image_id': [x.split('/')[-1].split('.')[0] for x in image_paths], 'PredictionString': sub_strings})
    return sub, sub_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
    )
    parser.add_argument(
        '--resolution', default=1024, type=int
    )
    hyperparams = parser.parse_args()

    model_name = hyperparams.model_name
    resolution = hyperparams.resolution

    os.makedirs('subs/{}_{}'.format(model_name, resolution), exist_ok=True)

    for fold in range(5):
        preds = pd.read_pickle('pkl_preds/{}/test/fold{}.pkl'.format(model_name, fold))
        original_test = pd.read_csv('data/test.csv', sep=',')
        with open('data/folds/coco_test.json', 'r') as file:
            image_paths = [x['file_name'].split('/')[-1] for x in json.load(file)['images']]

        preds = np.array(preds)
        print(preds.shape)

        sub, sub_strings = create_sub(preds, image_paths, original_test, resolution)
        sub.to_csv('subs/{}_{}/fold{}.csv'.format(model_name, resolution, fold), index=False)
