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


OUTPUT_PATH_YOLO = INPUT_PATH + 'yolo/'
if not os.path.isdir(OUTPUT_PATH_YOLO):
    os.mkdir(OUTPUT_PATH_YOLO)

OUTPUT_PATH_YOLO_FULL = INPUT_PATH + 'yolo_full/'
if not os.path.isdir(OUTPUT_PATH_YOLO_FULL):
    os.mkdir(OUTPUT_PATH_YOLO_FULL)


def draw_boxes(image_id, boxes, scores, labels):
    image = read_single_image(INPUT_PATH + 'train_png_div_2/{}.png'.format(image_id))
    print(image.shape, image.min(), image.max())
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


def copy_images_for_yolo(type='small'):
    if type == 'small':
        files = glob.glob(INPUT_PATH + 'train_png_div_2/*.png')
        out_path = OUTPUT_PATH_YOLO + 'images/'
    else:
        files = glob.glob(INPUT_PATH + 'train_png/*.png')
        out_path = OUTPUT_PATH_YOLO_FULL + 'images/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = out_path + 'train/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for f in files:
        shutil.copy(f, out_path + os.path.basename(f))


def create_labels_for_yolo(split_file, type='small'):
    classes = get_classes_array()
    if type == 'small':
        out_path = OUTPUT_PATH_YOLO + 'labels/'
    else:
        out_path = OUTPUT_PATH_YOLO_FULL + 'labels/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = out_path + 'train/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    s = pd.read_csv(split_file)
    sizes = get_train_test_image_sizes()

    groupby = s.groupby('id')
    for image_id, group in groupby:
        print(image_id, len(group))
        out_file_path = out_path + '{}.txt'.format(image_id)
        out = open(out_file_path, 'w')
        for _, row in group.iterrows():
            if str(row['x1']) == 'nan':
                continue
            x1 = row['x1'] / sizes[image_id][1]
            y1 = row['y1'] / sizes[image_id][0]
            x2 = row['x2'] / sizes[image_id][1]
            y2 = row['y2'] / sizes[image_id][0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            out.write('{} {} {} {} {}\n'.format(classes.index(row['class']), center_x, center_y, width, height))
        out.close()


def create_split_for_yolo5(split_file, out_folder, type='small'):
    classes = get_classes_array()[:-1]
    if type == 'small':
        training_img_directory = OUTPUT_PATH_YOLO + 'images/train/'
    else:
        training_img_directory = OUTPUT_PATH_YOLO_FULL + 'images/train/'
    KFOLD_SPLIT_FILE = OUTPUT_PATH + 'kfold_split_5_42.csv'
    s = pd.read_csv(KFOLD_SPLIT_FILE, dtype={'image_id': np.str})
    folds = s['fold'].max() + 1
    data = pd.read_csv(split_file)

    for fold_num in range(folds):
        part = s[s['fold'] != fold_num].copy()
        train_ids = part['image_id'].values
        train_df = data[data['id'].isin(train_ids)]
        out_path_train = out_folder + 'fold_{}_train.txt'.format(fold_num)
        train_df['id'] = training_img_directory + train_df['id'] + '.png'
        out = open(out_path_train, 'w')
        for id in train_df['id'].values:
            out.write('{}\n'.format(id))
        out.close()

        part = s[s['fold'] == fold_num].copy()
        valid_ids = part['image_id'].values
        valid_df = data[data['id'].isin(valid_ids)]
        out_path_valid = out_folder + 'fold_{}_valid.csv'.format(fold_num)
        valid_df['id'] = training_img_directory + valid_df['id'] + '.png'
        out = open(out_path_valid, 'w')
        for id in valid_df['id'].values:
            out.write('{}\n'.format(id))
        out.close()

        # Create XML
        out = open(out_folder + 'fold_{}.xml'.format(fold_num), 'w')
        out.write('train: {}\n'.format(out_path_train))
        out.write('val: {}\n'.format(out_path_valid))
        out.write('nc: {}\n'.format(14))
        out.write('names: {}\n'.format(list(classes)))
        out.close()


def create_fold_folders(fold_num):
    files = glob.glob(OUTPUT_PATH_YOLO + 'images/train/*.png')
    out_fold_folder = OUTPUT_PATH_YOLO + 'images/train_fold{}/'.format(fold_num)
    if not os.path.isdir(out_fold_folder):
        os.mkdir(out_fold_folder)

    print(len(files))
    KFOLD_SPLIT_FILE = OUTPUT_PATH + 'kfold_split_5_42.csv'
    s = pd.read_csv(KFOLD_SPLIT_FILE, dtype={'image_id': np.str})
    part = s[s['fold'] == fold_num].copy()
    fold_ids = set(part['image_id'].values)
    print(len(fold_ids))
    for f in files:
        file_id = os.path.basename(f)[:-4]
        if file_id in fold_ids:
            shutil.copy(f, out_fold_folder + file_id + '.png')


def copy_test_images_for_yolo(type='small'):
    if type == 'small':
        files = glob.glob(INPUT_PATH + 'test_png_div_4/*.png')
        out_path = OUTPUT_PATH_YOLO + 'images/'
    else:
        files = glob.glob(INPUT_PATH + 'test_png/*.png')
        out_path = OUTPUT_PATH_YOLO_FULL + 'images/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = out_path + 'test/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for f in files:
        shutil.copy(f, out_path + os.path.basename(f))


if __name__ == '__main__':
    split_file = OUTPUT_PATH + 'boxes_description_iou_0.4_div_1.csv'
    out_folder = OUTPUT_PATH + 'yolo5_data/'
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    copy_images_for_yolo('small')
    copy_test_images_for_yolo('small')
    create_labels_for_yolo(split_file, 'small')
    create_split_for_yolo5(split_file, out_folder, type='small')
    for i in range(5):
        create_fold_folders(i)

    if 0:
        out_folder_full = OUTPUT_PATH + 'yolo5_data_full/'
        if not os.path.isdir(out_folder_full):
            os.mkdir(out_folder_full)

        copy_images_for_yolo('full')
        copy_test_images_for_yolo('full')
        create_labels_for_yolo(split_file, 'full')
        create_split_for_yolo5(split_file, out_folder_full, type='full')
