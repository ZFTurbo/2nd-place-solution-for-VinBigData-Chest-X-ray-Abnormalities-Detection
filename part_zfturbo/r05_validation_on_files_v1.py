# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from map_boxes import mean_average_precision_for_boxes


def prepare_annotations(ann):
    sizes = get_train_test_image_sizes()
    classes = get_classes_array()
    a = []
    for index, row in ann.iterrows():
        id, x1, y1, x2, y2, label, score = row['id'], row['x1'], row['y1'], row['x2'], row['y2'], row['class'], row['score']
        x1 /= sizes[id][1]
        y1 /= sizes[id][0]
        x2 /= sizes[id][1]
        y2 /= sizes[id][0]
        if label in classes:
            label = str(classes.index(label))
        else:
            label = '14'
            x1 = 0
            y1 = 0
            x2 = 1
            y2 = 1
        a.append([id, label, x1, x2, y1, y2])
    return a


def prepare_predictions(pr):
    sizes = get_train_test_image_sizes()
    a = []
    for index, row in pr.iterrows():
        id, pred_str = row['image_id'], row['PredictionString']
        if str(pred_str) == 'nan':
            continue
        arr = pred_str.strip().split(' ')
        for k in range(0, len(arr), 6):
            cls = str(arr[k])
            prob = float(arr[k + 1])
            x1 = float(arr[k + 2]) / sizes[id][1]
            y1 = float(arr[k + 3]) / sizes[id][0]
            x2 = float(arr[k + 4]) / sizes[id][1]
            y2 = float(arr[k + 5]) / sizes[id][0]
            if cls == '14':
                a.append([id, cls, prob, 0, 1, 0, 1])
            else:
                a.append([id, cls, prob, x1, x2, y1, y2])

    return a


def validate_v1(annotations, predictions, verbose=True):
    thr_value = 0.4
    num_classes = 15
    if type(annotations) is str:
        ann = pd.read_csv(annotations)
    else:
        ann = annotations
    if type(predictions) is str:
        pr = pd.read_csv(predictions)
    else:
        pr = predictions

    # Fix different IDs
    unique_preds = pr['image_id'].unique()
    if verbose:
        print('Prediction Ids: {}'.format(len(unique_preds)))
    unique_ann = ann['id'].unique()
    if verbose:
        print('Annotation Ids: {}'.format(len(unique_ann)))
    ann = ann[ann['id'].isin(unique_preds)]
    unique_ann = ann['id'].unique()
    if verbose:
        print('Reduced annotation Ids: {}'.format(len(unique_ann)))

    ann = prepare_annotations(ann)
    pr = prepare_predictions(pr)
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, pr, iou_threshold=thr_value, verbose=verbose)
    map = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        try:
            if verbose:
                print('Class: {:2d} Entries: {:5d} AP: {:.6f}'.format(i, int(average_precisions[str(i)][1]), average_precisions[str(i)][0]))
            map[i] = average_precisions[str(i)][0]
        except Exception as e:
            if verbose:
                print('No class found: {}'.format(i))
            map[i] = 0
    map_no_last_class = map[:-1].mean()
    if verbose:
        print('mAP value: {:.6f}'.format(mean_ap))
        print('mAP value no last class: {:.6f}'.format(map_no_last_class))
    return map_no_last_class


if __name__ == '__main__':
    if 1:
        predictions = SUBM_PATH + 'retina_resnet101_fold_0_0.1817_05_iou_0.3_thr_0.01_iou_0.25_train.csv'
        # predictions = SUBM_PATH + 'retinanet_resnet152_v2_validation_thr_0.01_iou_0.55_type_avg.csv'
        annotations = OUTPUT_PATH + 'boxes_description_iou_0.4_div_1_removed_rads.csv'
        # annotations = OUTPUT_PATH + 'valid/boxes_description_div_1_rad_id_R17.csv'
        # annotations = pd.read_csv(annotations)
        # predictions = pd.read_csv(predictions)
        validate_v1(annotations, predictions)


"""
Class:  0 Entries:   641 AP: 0.856116
Class:  1 Entries:    56 AP: 0.152233
Class:  2 Entries:   126 AP: 0.089635
Class:  3 Entries:   453 AP: 0.905115
Class:  4 Entries:    72 AP: 0.302407
Class:  5 Entries:   151 AP: 0.175065
Class:  6 Entries:   175 AP: 0.199781
Class:  7 Entries:   390 AP: 0.194911
Class:  8 Entries:   359 AP: 0.123628
Class:  9 Entries:   361 AP: 0.035479
Class: 10 Entries:   329 AP: 0.460545
Class: 11 Entries:   765 AP: 0.199554
Class: 12 Entries:    26 AP: 0.105546
Class: 13 Entries:   603 AP: 0.217022
Class: 14 Entries:  2121 AP: 0.792756
mAP value: 0.320653
"""