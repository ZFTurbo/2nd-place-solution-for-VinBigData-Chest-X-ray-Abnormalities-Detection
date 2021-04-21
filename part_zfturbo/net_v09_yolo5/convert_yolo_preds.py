# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *


def convert_preds(fold_num, input_dir, out_file):
    sizes = get_train_test_image_sizes()
    s = pd.read_csv(OUTPUT_PATH + 'kfold_split_5_42.csv')
    files = glob.glob(input_dir + '*.txt')
    print(len(files))
    part = s[s['fold'] == fold_num].copy()
    print(len(part))
    valid_ids = set(part['image_id'].values)
    out = open(out_file, 'w')
    out.write('image_id,PredictionString\n')
    for f in files:
        image_id = os.path.basename(f)[:-4]
        in1 = open(f, 'r')
        lines = in1.readlines()
        in1.close()
        valid_ids.remove(image_id)
        out.write('{},'.format(image_id))
        pred_str = ''
        for line in lines:
            arr = line.strip().split(' ')
            class_id = arr[0]
            x = float(arr[1])
            y = float(arr[2])
            w = float(arr[3])
            h = float(arr[4])
            x1 = x - (w / 2)
            x2 = x + (w / 2)
            y1 = y - (h / 2)
            y2 = y + (h / 2)
            conf = arr[5]

            x1 = int(round(x1 * sizes[image_id][1]))
            y1 = int(round(y1 * sizes[image_id][0]))
            x2 = int(round(x2 * sizes[image_id][1]))
            y2 = int(round(y2 * sizes[image_id][0]))

            pred_str += '{} {} {} {} {} {} '.format(class_id, conf, x1, y1, x2, y2)
        out.write('{}\n'.format(pred_str))

    print(len(valid_ids))

    # Output empty IDs
    for image_id in list(valid_ids):
        out.write('{},14 1 0 0 1 1\n'.format(image_id))

    out.close()


def convert_preds_test(input_dir, out_file):
    sizes = get_train_test_image_sizes()
    s = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    files = glob.glob(input_dir + '*.txt')
    print(len(files))
    part = s
    print(len(part))
    valid_ids = set(part['image_id'].values)
    out = open(out_file, 'w')
    out.write('image_id,PredictionString\n')
    for f in files:
        image_id = os.path.basename(f)[:-4]
        in1 = open(f, 'r')
        lines = in1.readlines()
        in1.close()
        valid_ids.remove(image_id)
        out.write('{},'.format(image_id))
        pred_str = ''
        for line in lines:
            arr = line.strip().split(' ')
            class_id = arr[0]
            x = float(arr[1])
            y = float(arr[2])
            w = float(arr[3])
            h = float(arr[4])
            x1 = x - (w / 2)
            x2 = x + (w / 2)
            y1 = y - (h / 2)
            y2 = y + (h / 2)
            conf = arr[5]

            x1 = int(round(x1 * sizes[image_id][1]))
            y1 = int(round(y1 * sizes[image_id][0]))
            x2 = int(round(x2 * sizes[image_id][1]))
            y2 = int(round(y2 * sizes[image_id][0]))

            pred_str += '{} {} {} {} {} {} '.format(class_id, conf, x1, y1, x2, y2)
        out.write('{}\n'.format(pred_str))

    print(len(valid_ids))

    # Output empty IDs
    for image_id in list(valid_ids):
        out.write('{},14 1 0 0 1 1\n'.format(image_id))

    out.close()


if __name__ == '__main__':
    for fold_num in range(5):
        input_dir = OUTPUT_PATH + 'yolov5_fold{}/valid_iou_0.25_0.01/labels/'.format(fold_num)
        out_file = SUBM_PATH + 'yolov5_fold{}_iou_0.25_thr_0.01_train.csv'.format(fold_num)
        # convert_preds(fold_num, input_dir, out_file)

        input_dir = OUTPUT_PATH + 'yolov5_fold{}/test_iou_0.25_0.01/labels/'.format(fold_num)
        out_file = SUBM_PATH + 'yolov5_fold{}_iou_0.25_thr_0.01_test.csv'.format(fold_num)
        convert_preds_test(input_dir, out_file)

        input_dir = OUTPUT_PATH + 'yolov5_fold{}/valid_iou_0.25_0.01_mirror/labels/'.format(fold_num)
        out_file = SUBM_PATH + 'yolov5_fold{}_iou_0.25_thr_0.01_mirror_train.csv'.format(fold_num)
        # convert_preds(fold_num, input_dir, out_file)

        input_dir = OUTPUT_PATH + 'yolov5_fold{}/test_iou_0.25_0.01_mirror/labels/'.format(fold_num)
        out_file = SUBM_PATH + 'yolov5_fold{}_iou_0.25_thr_0.01_mirror_test.csv'.format(fold_num)
        convert_preds_test(input_dir, out_file)
