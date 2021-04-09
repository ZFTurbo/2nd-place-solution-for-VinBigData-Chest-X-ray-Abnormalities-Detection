import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import shutil as sh

CUR_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
print('CUR_PATH', CUR_PATH)

image_id_column = 'image_id'
fold_column = 'fold'
label_column = 'class_id'
rad_id_column = 'rad_id'

original_boxes_df = pd.read_csv(f'{CUR_PATH}train.csv')
empty_images_ids = set(original_boxes_df.loc[original_boxes_df[label_column] == 14, image_id_column].unique())
print(len(empty_images_ids), 'original images without boxes')

major_rads = set(original_boxes_df.loc[original_boxes_df[rad_id_column] == 'R8', image_id_column].unique()) \
            | set(original_boxes_df.loc[original_boxes_df[rad_id_column] == 'R9', image_id_column].unique()) \
            | set(original_boxes_df.loc[original_boxes_df[rad_id_column] == 'R10', image_id_column].unique())
major_rads = major_rads - empty_images_ids
print('major rads images', len(major_rads))

hard_negative_images_df = pd.read_csv(f'{CUR_PATH}hard_empty_images.csv')
hard_negative_images = set(hard_negative_images_df[image_id_column].values)
print(len(hard_negative_images), 'hard negative images')

train_images_df = pd.read_csv(f'{CUR_PATH}roma_kfold_split_5_42.csv')

label_dir = f'{CUR_PATH}yolo_labels_sep_rads/'
image_dir = f'{CUR_PATH}vinbigdata/train/'
convertor_folder = f'{CUR_PATH}converter_stage2'

for fold in range(5):
    images_val = train_images_df.loc[train_images_df[fold_column] == fold, image_id_column].values
    for image_id in tqdm(original_boxes_df[image_id_column].unique()):
        # Skip some rads
        if image_id in major_rads: continue

        if image_id in images_val:
            path2save = 'val/'
        else:
            # Skip simple empty images for quicker training
            if (image_id in empty_images_ids) and (image_id not in hard_negative_images): continue
            path2save = 'train/'

        target_folder = f'{convertor_folder}/{fold}/'
        labels_folder = f'{target_folder}/labels/{path2save}'
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
        images_folder = f'{target_folder}/images/{path2save}'
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        image_data = original_boxes_df[original_boxes_df[image_id_column] == image_id]
        rads = image_data['rad_id'].unique()
        is_empty = image_data.iloc[0][label_column] == 14
        for rad_id in rads:
            file_name_prefix = f'{image_id}_{rad_id}'
            source_label_file = os.path.join(label_dir, file_name_prefix + '.txt')
            sh.copy(source_label_file, labels_folder)

            source_image_file = os.path.join(image_dir, image_id + '.png')
            dest_image_file = os.path.join(images_folder, file_name_prefix + '.png')
            sh.copy(source_image_file, dest_image_file)
            if is_empty:
                break
