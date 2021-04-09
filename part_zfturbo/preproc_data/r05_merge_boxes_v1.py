# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
import pydicom
from ensemble_boxes import weighted_boxes_fusion


def draw_boxes(image_id, boxes, scores, labels):
    image = read_single_image(INPUT_PATH + 'train_png_16bit_div_4/{}.png'.format(image_id))
    print(image.shape, image.min(), image.max())
    image = (image // 256).astype(np.uint8)
    boxes = np.array(boxes)
    boxes[:, 0] *= image.shape[1]
    boxes[:, 1] *= image.shape[0]
    boxes[:, 2] *= image.shape[1]
    boxes[:, 3] *= image.shape[0]
    boxes = np.round(boxes).astype(np.int32)
    for i, b in enumerate(boxes):
        thickness = int(round(2*scores[i]))
        color = get_color(int(labels[i]))
        image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color=color, thickness=thickness)
    show_resized_image_16bit(image)


def merge_boxes_center_net(div_scale=4, iou_same=0.4):
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    # train = train[train['image_id'].isin(['14600a97b1c302343b1b5850ed53ae13', '299278f67dc5e40ee4fd003595c6e8d7'])]
    unique_images = train['image_id'].unique()
    print(len(train), len(unique_images))
    cls_names = get_classes_array()

    sizes = dict()
    sizes_df = pd.read_csv(OUTPUT_PATH + 'image_width_height_train.csv')
    for index, row in sizes_df.iterrows():
        sizes[row['image_id']] = (row['height'], row['width'])

    groupby = train.groupby('image_id')
    out = open(OUTPUT_PATH + 'boxes_description_iou_{}_div_{}.csv'.format(iou_same, div_scale), 'w')
    out.write('id,x1,y1,x2,y2,class,score\n')

    for index, group in groupby:
        print(index, len(group))
        is_empty = True
        for _, row in group.iterrows():
            if row['class_id'] != 14:
                is_empty = False
                break
        if is_empty:
            out.write('{},,,,,,\n'.format(index))
            continue

        boxes = []
        scores = []
        labels = []
        # x_min,y_min,x_max,y_max
        for _, row in group.iterrows():
            if row['class_id'] == 14:
                continue
            image_id = row['image_id']
            x1 = row['x_min'] / sizes[image_id][1]
            y1 = row['y_min'] / sizes[image_id][0]
            x2 = row['x_max'] / sizes[image_id][1]
            y2 = row['y_max'] / sizes[image_id][0]

            if x2 < x1:
                print('Strange x2 < x1')
                exit()
            if y2 < y1:
                print('Strange y2 < y1')
                exit()
            if x2 > 1:
                print('Strange x2 > 1')
                exit()
            if y2 > 1:
                print('Strange y2 > 1')
                exit()

            boxes.append([x1, y1, x2, y2])
            labels.append(row['class_id'])
            scores.append(1.0)

        print(len(boxes), len(labels), len(scores))
        # print(boxes)
        # print(labels)
        # print(scores)
        # draw_boxes(index, boxes, scores, labels)
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=iou_same, weights=None, allows_overflow=True)
        print(len(boxes), len(labels), len(scores))
        # print(boxes)
        # print(labels)
        # print(scores)
        # draw_boxes(index, boxes, scores, labels)

        # Div 4 because I plan to use reduced images!
        scale_y = sizes[index][0] // div_scale
        scale_x = sizes[index][1] // div_scale

        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        boxes = np.round(boxes).astype(np.int32)
        labels = labels.astype(np.int32)

        for i in range(len(boxes)):
            if scores[i] > 3:
                scores[i] = 3

            out.write("{},{},{},{},{},{},{:.0f}\n".format(
                index,
                boxes[i, 0],
                boxes[i, 1],
                boxes[i, 2],
                boxes[i, 3],
                cls_names[labels[i]],
                scores[i]
            ))

    out.close()


def create_split_for_centernet(split_file, training_img_directory, out_folder):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    KFOLD_SPLIT_FILE = OUTPUT_PATH + 'kfold_split_5_42.csv'
    s = pd.read_csv(KFOLD_SPLIT_FILE, dtype={'image_id': np.str})
    folds = s['fold'].max() + 1
    data = pd.read_csv(split_file)

    for fold_num in range(folds):
        part = s[s['fold'] != fold_num].copy()
        train_ids = part['image_id'].values
        train_df = data[data['id'].isin(train_ids)]
        out_path = out_folder + 'fold_{}_train.csv'.format(fold_num)
        train_df['id'] = training_img_directory + train_df['id'] + '.png'
        train_df.to_csv(out_path, index=False, float_format='%.12g')

        part = s[s['fold'] == fold_num].copy()
        valid_ids = part['image_id'].values
        valid_df = data[data['id'].isin(valid_ids)]
        out_path = out_folder + 'fold_{}_valid.csv'.format(fold_num)
        valid_df['id'] = training_img_directory + valid_df['id'] + '.png'
        valid_df.to_csv(out_path, index=False, float_format='%.12g')

    CLASSES = get_classes_array()
    out = open(out_folder + 'classes.txt', 'w')
    for i, c in enumerate(CLASSES):
        if i == 14:
            continue
        out.write('{},{}\n'.format(c, i))
    out.close()


def check_max_objects(split_file):
    data = pd.read_csv(split_file)
    print(data['id'].value_counts())


def create_test_file_centernet(testing_img_directory, out_folder):
    out = open(out_folder + 'test.csv', 'w')
    # out.write('id,x1,y1,x2,y2,class,score\n')
    files = glob.glob(testing_img_directory + '*.png')
    for f in files:
        id = os.path.basename(f)[:-4]
        out.write('{},,,,,,\n'.format(f))
    out.close()


def remove_empty_images(in_folder, out_folder):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    files = glob.glob(in_folder + '*.csv')
    for f in files:
        print(f)
        s = pd.read_csv(f)
        print(len(s))
        s = s[~s['class'].isna()]
        print(len(s))
        s['x1'] = s['x1'].astype(np.int32)
        s['x2'] = s['x2'].astype(np.int32)
        s['y1'] = s['y1'].astype(np.int32)
        s['y2'] = s['y2'].astype(np.int32)
        s.to_csv(out_folder + os.path.basename(f), index=False)


if __name__ == '__main__':
    merge_boxes_center_net(div_scale=1, iou_same=0.4)
    merge_boxes_center_net(div_scale=2, iou_same=0.4)
    split_file = OUTPUT_PATH + 'boxes_description_iou_0.4_div_2.csv'
    out_folder = OUTPUT_PATH + 'retinanet_div_2/'
    training_img_directory = INPUT_PATH + 'train_png_div_2/'
    create_split_for_centernet(split_file, training_img_directory, out_folder)
    testing_img_directory = INPUT_PATH + 'test_png_div_2/'
    create_test_file_centernet(testing_img_directory, out_folder)
