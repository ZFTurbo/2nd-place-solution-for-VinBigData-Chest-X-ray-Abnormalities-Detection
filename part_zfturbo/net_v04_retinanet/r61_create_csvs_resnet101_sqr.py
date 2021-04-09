# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def find_matching_box(boxes_list, new_box, match_iou=0.7):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def merge_boxes_v2(box1, box2, w1, w2, type):
    box = [-1, -1, -1, -1, -1, -1]
    box[0] = box1[0]
    if type == 'avg':
        box[1] = ((w1 * box1[1]) + (w2 * box2[1])) / (w1 + w2)
    elif type == 'max':
        box[1] = max(box1[1], box2[1])
    elif type == 'mul':
        box[1] = np.sqrt(box1[1]*box2[1])
    else:
        exit()
    box[2] = (w1*box1[2] + w2*box2[2]) / (w1 + w2)
    box[3] = (w1*box1[3] + w2*box2[3]) / (w1 + w2)
    box[4] = (w1*box1[4] + w2*box2[4]) / (w1 + w2)
    box[5] = (w1*box1[5] + w2*box2[5]) / (w1 + w2)
    return box


def merge_all_boxes_for_image(boxes, intersection_thr=0.5, type='avg'):

    new_boxes = boxes[0].copy()
    init_weight = 1/len(boxes)
    weights = [init_weight] * len(new_boxes)

    for j in range(1, len(boxes)):
        for k in range(len(boxes[j])):
            index, best_iou = find_matching_box(new_boxes, boxes[j][k], intersection_thr)
            if index != -1:
                new_boxes[index] = merge_boxes_v2(new_boxes[index], boxes[j][k], weights[index], init_weight, type)
                weights[index] += init_weight
            else:
                new_boxes.append(boxes[j][k])
                weights.append(init_weight)

    for i in range(len(new_boxes)):
        new_boxes[i][1] *= weights[i]
    return np.array(new_boxes)


def filter_boxes(boxes, scores, labels, thr):
    new_boxes = []
    for i in range(boxes.shape[0]):
        box = []
        for j in range(boxes.shape[1]):
            label = labels[i, j].astype(np.int64)
            score = scores[i, j]
            if score < thr:
                break
            # Mirror fix !!!
            if i % 2 == 0:
                b = [int(label), float(score), float(boxes[i, j, 0]), float(boxes[i, j, 1]), float(boxes[i, j, 2]), float(boxes[i, j, 3])]
            else:
                b = [int(label), float(score), 1 - float(boxes[i, j, 2]), float(boxes[i, j, 1]), 1 - float(boxes[i, j, 0]), float(boxes[i, j, 3])]
            box.append(b)
        new_boxes.append(box)
    return new_boxes


def create_csv_for_retinanet_predictions_v2_format_v1(
        input_dir,
        out_file,
        skip_box_thr=0.05,
        intersection_thr=0.5,
        limit_boxes=300,
        type='avg'
):
    classes = get_classes_array()
    out = open(out_file, 'w')
    out.write('image_id,conf,x1,y1,x2,y2,label\n')
    files = glob.glob(input_dir + '*.pkl')
    for f in files:
        id = os.path.basename(f)[:-4]
        boxes, scores, labels = load_from_file_fast(f)
        filtered_boxes = filter_boxes(boxes, scores, labels, skip_box_thr)
        # print(len(filtered_boxes[0]), len(filtered_boxes[1]))
        # print(filtered_boxes[0], filtered_boxes[1])
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, intersection_thr, type)
        # reduced_boxes = reduce_similar(merged_boxes)
        print(id, len(filtered_boxes[0]), len(filtered_boxes[1]), len(merged_boxes))
        if len(merged_boxes) > limit_boxes:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        for i in range(len(merged_boxes)):
            label = int(merged_boxes[i][0])
            score = merged_boxes[i][1]
            b = merged_boxes[i][2:]

            google_name = 'wheat'

            xmin = b[0]
            if xmin < 0:
                xmin = 0
            if xmin > 1:
                xmin = 1

            xmax = b[2]
            if xmax < 0:
                xmax = 0
            if xmax > 1:
                xmax = 1

            ymin = b[1]
            if ymin < 0:
                ymin = 0
            if ymin > 1:
                ymin = 1

            ymax = b[3]
            if ymax < 0:
                ymax = 0
            if ymax > 1:
                ymax = 1

            if (xmax < xmin):
                print('X min value larger than max value {}: {} {}'.format('wheat', xmin, xmax))
                continue

            if (ymax < ymin):
                print('Y min value larger than max value {}: {} {}'.format('wheat', ymin, ymax))
                continue

            if abs(xmax - xmin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format('wheat', xmin, xmax))
                continue

            if abs(ymax - ymin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format('wheat', ymin, ymax))
                continue

            str1 = "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(id, score, xmin, ymin, xmax, ymax, label)
            out.write(str1)
        out.write('\n')


def create_csv_for_retinanet_predictions(
        input_dir,
        out_file,
        skip_box_thr=0.05,
        intersection_thr=0.5,
        limit_boxes=300,
        type='avg'
):
    verbose = False
    classes = get_classes_array()
    sizes = get_train_test_image_sizes()
    out = open(out_file, 'w')
    out.write('image_id,PredictionString\n')
    files = glob.glob(input_dir + '*.pkl')
    for f in files:
        id = os.path.basename(f)[:-4]
        boxes, scores, labels = load_from_file_fast(f)
        filtered_boxes = filter_boxes(boxes, scores, labels, skip_box_thr)
        # print(len(filtered_boxes[0]), len(filtered_boxes[1]))
        # print(filtered_boxes[0], filtered_boxes[1])
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, intersection_thr, type)
        # reduced_boxes = reduce_similar(merged_boxes)
        if verbose:
            print(id, len(filtered_boxes[0]), len(filtered_boxes[1]), len(merged_boxes))
        if len(merged_boxes) > limit_boxes:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        out.write("{},".format(id))
        if len(merged_boxes) > 0:
            for i in range(len(merged_boxes)):
                label = int(merged_boxes[i][0])
                score = merged_boxes[i][1]
                b = merged_boxes[i][2:]

                xmin = b[0]
                if xmin < 0:
                    xmin = 0
                if xmin > 1:
                    xmin = 1

                xmax = b[2]
                if xmax < 0:
                    xmax = 0
                if xmax > 1:
                    xmax = 1

                ymin = b[1]
                if ymin < 0:
                    ymin = 0
                if ymin > 1:
                    ymin = 1

                ymax = b[3]
                if ymax < 0:
                    ymax = 0
                if ymax > 1:
                    ymax = 1

                if (xmax < xmin):
                    print('X min value larger than max value {}: {} {}'.format('label', xmin, xmax))
                    continue

                if (ymax < ymin):
                    print('Y min value larger than max value {}: {} {}'.format('label', ymin, ymax))
                    continue

                if abs(xmax - xmin) < 1e-5:
                    print('Too small diff for {}: {} and {}'.format('label', xmin, xmax))
                    continue

                if abs(ymax - ymin) < 1e-5:
                    print('Too small diff for {}: {} and {}'.format('label', ymin, ymax))
                    continue

                xmin = int(round(xmin * sizes[id][1]))
                xmax = int(round(xmax * sizes[id][1]))
                ymin = int(round(ymin * sizes[id][0]))
                ymax = int(round(ymax * sizes[id][0]))
                str1 = "{} {:.6f} {} {} {} {} ".format(label, score, xmin, ymin, xmax, ymax)
                out.write(str1)
        else:
            str1 = "14 1 0 0 1 1"
            out.write(str1)
        out.write('\n')


if __name__ == '__main__':
    from r05_validation_on_files_v1 import validate_v1
    skip_box_thr = 0.05
    intersection_thr = 0.55
    limit_boxes = 100
    type = 'avg'

    annotations = OUTPUT_PATH + 'boxes_description_iou_0.4_div_1.csv'

    out_folders = [
        OUTPUT_PATH + 'resnet101_fold_0_0.3573_26_iou_0.3_valid/',
        OUTPUT_PATH + 'resnet101_fold_1_0.3481_35_iou_0.3_valid/',
        OUTPUT_PATH + 'resnet101_fold_2_0.3804_24_iou_0.3_valid/',
        OUTPUT_PATH + 'resnet101_fold_3_0.3584_24_iou_0.3_valid/',
        OUTPUT_PATH + 'resnet101_fold_4_0.3514_12_iou_0.3_valid/',
    ]

    best_params_dict = dict()
    if 1:
        for o in out_folders:
            params = dict()
            best_params = -1
            best_score = -1
            for skip_box_thr in [0.09, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]:
                for intersection_thr in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
                    start_time = time.time()
                    out_file = SUBM_PATH + 'retina_' + os.path.basename(os.path.dirname(o))[:-6] + '_thr_{}_iou_{}_train.csv'.format(skip_box_thr, intersection_thr)
                    create_csv_for_retinanet_predictions(
                        o,
                        out_file,
                        skip_box_thr,
                        intersection_thr,
                        limit_boxes,
                        type=type
                    )
                    map = validate_v1(annotations, out_file, verbose=False)
                    params[(skip_box_thr, intersection_thr)] = map
                    os.remove(out_file)
                    if map > best_score:
                        best_score = map
                        best_params = (skip_box_thr, intersection_thr)
                    print('Fold: {} Skip: {} Thr: {} mAP: {:.6f} Time: {:.2f} sec'.format(os.path.basename(os.path.dirname(o)), skip_box_thr, intersection_thr, map, time.time() - start_time))

            print(params)
            print('Out folder: {} Best score: {:.6f} Best params: {}'.format(os.path.basename(os.path.dirname(o)), best_score, best_params))
            best_params_dict[o] = best_params

    for o in out_folders:
        print('Go {} Params: {}'.format(os.path.basename(os.path.dirname(o)), best_params_dict[o]))
        params = dict()
        skip_box_thr = best_params_dict[o][0]
        intersection_thr = best_params_dict[o][1]
        out_file = SUBM_PATH + 'retina_' + os.path.basename(os.path.dirname(o))[
                                           :-6] + '_thr_{}_iou_{}_train.csv'.format(skip_box_thr,
                                                                                    intersection_thr)
        create_csv_for_retinanet_predictions(
            o,
            out_file,
            skip_box_thr,
            intersection_thr,
            limit_boxes,
            type=type
        )

    for o in out_folders:
        print('Go {} Params: {}'.format(os.path.basename(os.path.dirname(o)), best_params_dict[o]))
        skip_box_thr = best_params_dict[o][0]
        intersection_thr = best_params_dict[o][1]
        out_file = SUBM_PATH + 'retina_' + os.path.basename(os.path.dirname(o))[:-6] + '_thr_{}_iou_{}_test.csv'.format(skip_box_thr, intersection_thr)
        create_csv_for_retinanet_predictions(
            o[:-7] + '_test/',
            out_file,
            skip_box_thr,
            intersection_thr,
            limit_boxes,
            type=type
        )


"""
Out folder: resnet101_fold_0_0.3573_26_iou_0.3_valid Best score: 0.366694 Best params: (0.05, 0.35)
Out folder: resnet101_fold_1_0.3481_35_iou_0.3_valid Best score: 0.357482 Best params: (0.01, 0.45)
Out folder: resnet101_fold_2_0.3804_24_iou_0.3_valid Best score: 0.385898 Best params: (0.03, 0.45)
Out folder: resnet101_fold_3_0.3584_24_iou_0.3_valid Best score: 0.357226 Best params: (0.05, 0.45)
Out folder: resnet101_fold_4_0.3514_12_iou_0.3_valid Best score: 0.360805 Best params: (0.03, 0.45)
"""