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

train_df = pd.read_csv(f'{CUR_PATH}train.csv')
print(len(train_df[image_id_column].unique()), 'images')
print(len(train_df), 'boxes')

label_dir = f'{CUR_PATH}yolo_labels_sep_rads/'
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

#train_df = train_df[train_df.class_id!=14].reset_index(drop = True)
print(len(train_df[image_id_column].unique()), 'images')
print(len(train_df), 'boxes')

old_yolo_train_df = pd.read_csv(f'{CUR_PATH}train_with_size.csv')
train_meta_df = old_yolo_train_df[[image_id_column, 'width', 'height']].drop_duplicates()
print(train_meta_df.head())

train_df = train_df.merge(train_meta_df, on=image_id_column, how='left')

train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis =1)

train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis =1)

train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)

train_df['area'] = train_df['w']*train_df['h']
print(train_df.head())

for image_id in train_df[image_id_column].unique():
    image_data = train_df[train_df[image_id_column] == image_id]
    rads = image_data['rad_id'].unique()
    is_empty = False
    for rad_id in rads:
        data = image_data.loc[image_data['rad_id'] == rad_id, ['class_id', 'x_mid', 'y_mid', 'w', 'h']]
        file_name = f'{label_dir}{image_id}_{rad_id}.txt'
        with open(file_name, 'w+') as f:
            for idx in range(len(data)):
                row = data.iloc[idx]
                if int(row.class_id) == 14:
                    is_empty = True
                    continue
                text = f'{int(row.class_id)} {row.x_mid} {row.y_mid} {row.w} {row.h}'
                f.write(text)
                f.write("\n")
        if is_empty:
            break
