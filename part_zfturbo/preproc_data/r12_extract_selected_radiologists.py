# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
from preproc_data.r05_merge_boxes_v1 import create_split_for_centernet, create_test_file_centernet


def extract_boxes_for_rads(init_train_file):
    exclude_rads = ('R10', 'R8', 'R9')
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    groupby = train.groupby('image_id')

    exclude_ids = set()
    empty_images = []
    for index, group in groupby:
        u = tuple(sorted(group['rad_id'].unique()))
        class_ids = tuple(sorted(group['class_id'].unique()))
        print(index, u, class_ids)
        if u == exclude_rads:
            exclude_ids |= set([index])
        if class_ids == (14,):
            exclude_ids |= set([index])
            empty_images.append(index)

    print('Total exclude_ids: {}'.format(len(exclude_ids)))
    print('Old train: {}'.format(len(train)))
    train = train[(~train['image_id'].isin(exclude_ids)) | (train['image_id'].isin(empty_images))]
    # train = train[(~train['image_id'].isin(exclude_ids))]
    print('New train: {}'.format(len(train)))
    print(train['rad_id'].value_counts())
    # print(train[train['rad_id'] == 'R2'])
    print('IDs to use: {} Empty IDs: {}'.format(len(train['image_id'].unique()), len(empty_images)))

    s = pd.read_csv(init_train_file)
    for c in s.columns.values:
        try:
            s[c] = s[c].astype('Int64')
        except:
            pass
    part = s[s['id'].isin(train['image_id'].unique())]
    print(len(s), len(part))
    print(part['class'].value_counts())
    part.to_csv(init_train_file[:-4] + '_removed_rads.csv', index=False)


if __name__ == '__main__':
    init_train_file = OUTPUT_PATH + 'boxes_description_iou_0.4_div_2_sqr.csv'
    extract_boxes_for_rads(init_train_file)

    out_folder = OUTPUT_PATH + 'retinanet_train_sqr_data_removed_rads/'
    training_img_directory = INPUT_PATH + 'train_png_div_2/'
    testing_img_directory = INPUT_PATH + 'test_png_div_2/'

    create_split_for_centernet(init_train_file[:-4] + '_removed_rads.csv', training_img_directory, out_folder)
    create_test_file_centernet(testing_img_directory, out_folder)
