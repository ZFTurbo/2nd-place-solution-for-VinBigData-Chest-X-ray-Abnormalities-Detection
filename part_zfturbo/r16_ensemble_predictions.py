# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from ensemble_boxes import weighted_boxes_fusion


def ensemble(
    subm_list,
    iou_same=0.5,
    out_path=None,
    skip_box_thr=0.00000001,
):
    sizes = get_train_test_image_sizes()
    preds = []
    weights = []
    checker = None
    for path, weight in subm_list:
        s = pd.read_csv(path)
        s.sort_values('image_id', inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s['image_id']
        if checker:
            if tuple(ids) != checker:
                print(set(checker) - set(ids))
                print('Different IDS!', len(tuple(ids)), path)
                exit()
        else:
            checker = tuple(ids)
        preds.append(s['PredictionString'].values)
        weights.append(weight)

    if out_path is None:
        out_path = SUBM_PATH + 'ensemble_iou_{}.csv'.format(iou_same)
    out = open(out_path, 'w')
    out.write('image_id,PredictionString\n')
    for j, id in enumerate(list(checker)):
        # print(id)
        boxes_list = []
        scores_list = []
        labels_list = []
        empty = True
        for i in range(len(preds)):
            boxes = []
            scores = []
            labels = []
            p1 = preds[i][j]
            if str(p1) != 'nan':
                arr = p1.strip().split(' ')
                for k in range(0, len(arr), 6):
                    cls = int(arr[k])
                    prob = float(arr[k + 1])
                    x1 = float(arr[k + 2]) / sizes[id][1]
                    y1 = float(arr[k + 3]) / sizes[id][0]
                    x2 = float(arr[k + 4]) / sizes[id][1]
                    y2 = float(arr[k + 5]) / sizes[id][0]
                    boxes.append([x1, y1, x2, y2])
                    scores.append(prob)
                    labels.append(cls)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=iou_same,
            skip_box_thr=skip_box_thr,
            weights=weights,
            allows_overflow=True
        )
        # print(len(boxes), len(labels), len(scores))
        if len(boxes) == 0:
            out.write('{},14 1 0 0 1 1\n'.format(id, ))
        else:
            final_str = ''
            for i in range(len(boxes)):
                cls = int(labels[i])
                prob = scores[i]
                x1 = int(boxes[i][0] * sizes[id][1])
                y1 = int(boxes[i][1] * sizes[id][0])
                x2 = int(boxes[i][2] * sizes[id][1])
                y2 = int(boxes[i][3] * sizes[id][0])
                if cls == 14:
                    final_str += '{} {} {} {} {} {} '.format(cls, prob, 0, 0, 1, 1)
                else:
                    final_str += '{} {} {} {} {} {} '.format(cls, prob, x1, y1, x2, y2)
            out.write('{},{}\n'.format(id, final_str.strip()))

    out.close()
    return out_path


def get_test_from_subm_list(subm_list):
    out = []
    for s, w in subm_list:
        s1 = s.replace('_train', '_test')
        out.append((s1, w))
    return out


def ensemble_experiment_v4_yolo():

    sp = SUBM_PATH
    subm_list = [
        (sp + 'yolov5_fold0_iou_0.25_thr_0.01_train.csv', 1),
        (sp + 'yolov5_fold1_iou_0.25_thr_0.01_train.csv', 1),
        (sp + 'yolov5_fold2_iou_0.25_thr_0.01_train.csv', 1),
        (sp + 'yolov5_fold3_iou_0.25_thr_0.01_train.csv', 1),
        (sp + 'yolov5_fold4_iou_0.25_thr_0.01_train.csv', 1),
    ]
    subm_list_test = get_test_from_subm_list(subm_list)

    best_iou = 0.3
    out_path = SUBM_PATH + 'ensemble_yolo_standard.csv'.format(len(subm_list_test), best_iou)
    predictions = ensemble(subm_list_test, best_iou, out_path)


def ensemble_experiment_v12_yolo_mirror():
    sp = SUBM_PATH
    subm_list_test = [
        (sp + 'yolov5_fold0_iou_0.25_thr_0.01_mirror_test.csv', 1),
        (sp + 'yolov5_fold1_iou_0.25_thr_0.01_mirror_test.csv', 1),
        (sp + 'yolov5_fold2_iou_0.25_thr_0.01_mirror_test.csv', 1),
        (sp + 'yolov5_fold3_iou_0.25_thr_0.01_mirror_test.csv', 1),
        (sp + 'yolov5_fold4_iou_0.25_thr_0.01_mirror_test.csv', 1),
    ]

    best_iou = 0.3
    best_map = -1
    out_path = SUBM_PATH + 'ensemble_yolo_mirror.csv'.format(len(subm_list_test), best_iou, best_map)
    predictions = ensemble(subm_list_test, best_iou, out_path)


def ensemble_experiment_v13_ensemble_yolo():
    iou = 0.4
    mean_ap = -1
    subm_list = [
        (SUBM_PATH + 'ensemble_yolo_standard.csv', 1),
        (SUBM_PATH + 'ensemble_yolo_mirror.csv', 1),
    ]
    out_path = SUBM_PATH + 'ensemble_yolo_final.csv'.format(len(subm_list), iou, mean_ap)
    predictions = ensemble(subm_list, iou, out_path)


def ensemble_experiment_v17_retinanet_resnet101_sqr():

    sp = SUBM_PATH
    subm_list = [
        (sp + 'retina_resnet101_fold_0_0.3573_26_iou_0.3_thr_0.05_iou_0.35_test.csv', 1),
        (sp + 'retina_resnet101_fold_1_0.3481_35_iou_0.3_thr_0.01_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_2_0.3804_24_iou_0.3_thr_0.03_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_3_0.3584_24_iou_0.3_thr_0.05_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_4_0.3514_12_iou_0.3_thr_0.03_iou_0.45_test.csv', 1),
    ]
    subm_list_test = get_test_from_subm_list(subm_list)

    best_iou = 0.4
    skip_box_thr = 0.01
    out_path = SUBM_PATH + 'ensemble_retinanet_resnet101_sqr.csv'.format(len(subm_list_test), best_iou, skip_box_thr)
    predictions = ensemble(subm_list_test, best_iou, out_path, skip_box_thr)


def ensemble_experiment_v24_retinanet_resnet101_sqr_removed_radiologists():

    sp = SUBM_PATH
    subm_list = [
        (sp + 'retina_resnet101_fold_0_0.1817_05_iou_0.3_thr_0.03_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_1_0.2072_19_iou_0.3_thr_0.01_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_2_0.1938_03_iou_0.3_thr_0.01_iou_0.4_test.csv', 1),
        (sp + 'retina_resnet101_fold_3_0.1884_07_iou_0.3_thr_0.01_iou_0.45_test.csv', 1),
        (sp + 'retina_resnet101_fold_4_0.2227_05_iou_0.3_thr_0.01_iou_0.4_test.csv', 1),
    ]
    subm_list_test = get_test_from_subm_list(subm_list)

    best_iou = 0.4
    skip_box_thr = 0.01
    out_path = SUBM_PATH + 'ensemble_retinanet_resnet101_removed_rad.csv'.format(len(subm_list_test), best_iou, skip_box_thr)
    predictions = ensemble(subm_list_test, best_iou, out_path, skip_box_thr)


if __name__ == '__main__':
    ensemble_experiment_v4_yolo()
    ensemble_experiment_v12_yolo_mirror()
    ensemble_experiment_v13_ensemble_yolo()
    ensemble_experiment_v17_retinanet_resnet101_sqr()
    ensemble_experiment_v24_retinanet_resnet101_sqr_removed_radiologists()


"""
YOLO All 5 Folds IOU: 0.3 + Clasification = 0.245
YOLO All 5 mirror Folds IOU: 0.3 + Classification = 0.246
YOLO5 0.245 + 0.246 = 0.251 (IOU: 0.4)
ResNet101 SQR 5 folds IOU: 0.4: 0.246 - ensemble_experiment_v17_retinanet_resnet101_sqr()
ResNet101 SQR (Removed radiologists) 5 folds IOU: 0.4: 0.267 - ensemble_experiment_v24_retinanet_resnet101_sqr_removed_radiologists()
"""