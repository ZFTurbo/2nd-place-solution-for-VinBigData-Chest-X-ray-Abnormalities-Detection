import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

CUR_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
print('CUR_PATH', CUR_PATH)

import sys
sys.path.append(CUR_PATH + "yolo5")
from utils.datasets import *
from utils.general import *
import ytest  # import test.py to get mAP after each epoch
from ensemble_boxes import *
import argparse

import torch

image_id_column = 'image_id'

def image_rot(image, factor):
    return np.rot90(image, factor)

def flip_hor(image):
    return np.fliplr(image)

def bbox_rot90(bbox, factor, height, width):
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox (tuple): A bounding box tuple (x_min, y_min, x_max, y_max).
        factor (int): Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, width - x_max, y_max, width - x_min
    elif factor == 2:
        bbox = width - x_max, height - y_max, width - x_min, height - y_min
    elif factor == 3:
        bbox = height - y_max, x_min, height - y_min, x_max
    return bbox

def flip_hor_boxes(bbox, width):
    x_min, y_min, x_max, y_max = bbox[:4]
    return width - x_max, y_min, width - x_min, y_max

def image_tta0(image):
    return image

def box_reverse_tta0(box, height, width):
    return box

def image_tta1(image):
    return image_rot(image, 1)

def box_reverse_tta1(box, height, width):
    return bbox_rot90(box, 3, height, width)

def image_tta2(image):
    return image_rot(image, 2)

def box_reverse_tta2(box, height, width):
    return bbox_rot90(box, 2, height, width)

def image_tta3(image):
    return image_rot(image, 3)

def box_reverse_tta3(box, height, width):
    return bbox_rot90(box, 1, height, width)

def image_tta4(image):
    return flip_hor(image)

def box_reverse_tta4(box, height, width):
    return flip_hor_boxes(box, width)

def image_tta5(image):
    return image_rot(flip_hor(image), 1)

def box_reverse_tta5(box, height, width):
    rotated_box = bbox_rot90(box, 3, height, width)
    return flip_hor_boxes(rotated_box, height)

def image_tta6(image):
    return image_rot(flip_hor(image), 2)

def box_reverse_tta6(box, height, width):
    rotated_box = bbox_rot90(box, 2, height, width)
    return flip_hor_boxes(rotated_box, width)

def image_tta7(image):
    return image_rot(flip_hor(image), 3)

def box_reverse_tta7(box, height, width):
    rotated_box = bbox_rot90(box, 1, height, width)
    return flip_hor_boxes(rotated_box, height)

def get_tta_pair(ind):
    if ind == 0:
        return image_tta0, box_reverse_tta0
    if ind == 1:
        return image_tta1, box_reverse_tta1
    if ind == 2:
        return image_tta2, box_reverse_tta2
    if ind == 3:
        return image_tta3, box_reverse_tta3
    if ind == 4:
        return image_tta4, box_reverse_tta4
    if ind == 5:
        return image_tta5, box_reverse_tta5
    if ind == 6:
        return image_tta6, box_reverse_tta6
    if ind == 7:
        return image_tta7, box_reverse_tta7

def detect1Image(im0, imgsz, model, device, conf_thres, iou_thres):
    img = letterbox(im0, new_shape=imgsz)[0]
    # Convert
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0   
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    boxes = []
    scores = []
    labels = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(float(conf))
                labels.append(int(cls))

    return np.array(boxes), np.array(scores), np.array(labels)

def format_prediction_string(boxes, scores, labels):
    pred_strings = []
    if len(boxes) > 0 :
        for j in zip(labels, scores, boxes):
            pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))
    else:
        pred_strings.append("14 1 0 0 1 1")

    return " ".join(pred_strings)

def write_submission(test_images_ids, result_boxes, result_scores, result_labels, score_thr, submission_file_name):
    results = []
    for i, image in enumerate(test_images_ids):
        image_id = test_images_ids[i]
        cur_boxes = np.array(result_boxes[i])
        cur_scores = np.array(result_scores[i])
        cur_labels = np.array(result_labels[i])

        score_filter = cur_scores >= score_thr
        cur_boxes = cur_boxes[score_filter]
        cur_scores = cur_scores[score_filter]
        cur_labels = cur_labels[score_filter]
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(cur_boxes, cur_scores, cur_labels)
        }
        results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv(submission_file_name, index=False)

def merge_boxes_from_models(all_models_boxes, all_models_scores, all_models_labels, n_images, intersection_thr):
    n_variants = len(all_models_boxes)
    print(len(all_models_boxes), len(all_models_scores), len(all_models_labels))
    assert len(all_models_scores) == n_variants
    assert len(all_models_labels) == n_variants

    result_boxes = []
    result_scores = []
    result_labels = []
    for ii in range(n_images):
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        max_value = 10000
        for vi in range(n_variants):
            cur_pred_boxes = np.array(all_models_boxes[vi][ii], copy=False)
            cur_pred_scores = np.array(all_models_scores[vi][ii], copy=False)
            cur_pred_labels = np.array(all_models_labels[vi][ii], copy=False)

            # WBF expects the coordinates in 0-1 range.
            cur_pred_boxes = cur_pred_boxes / max_value

            pred_boxes.append(cur_pred_boxes)
            pred_scores.append(cur_pred_scores)
            pred_labels.append(cur_pred_labels)

        # Calculate WBF
        pred_boxes, pred_scores, pred_labels = weighted_boxes_fusion(
            pred_boxes,
            pred_scores,
            pred_labels,
            weights=None,
            iou_thr=intersection_thr,
            skip_box_thr=0
        )
        pred_boxes = np.round(pred_boxes * max_value).astype(int)

        assert len(pred_boxes) == len(pred_scores)
        #pred_boxes = make_boxes_int(pred_boxes)
        assert len(pred_boxes) == len(pred_scores)
        assert len(pred_boxes) == len(pred_labels)
        result_boxes.append(pred_boxes)
        result_scores.append(pred_scores)
        result_labels.append(pred_labels)
    return result_boxes, result_scores, result_labels

def predict_for_one_tta(weights, folder, imagenames, image_size, aug_ind):
    conf_thres = 0.01
    iou_thres = 0.4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    all_boxes = []
    all_scores = []
    all_labels = []

    image_tta, box_reverse_tta = get_tta_pair(aug_ind)

    for ind, name in enumerate(imagenames):
        if ind % 100 == 0:
            print('ind', ind)
        image_id = name.split('.')[0]
        original_image = cv2.imread('%s/%s.png' % (folder, image_id))  # BGR
        assert original_image is not None, 'Image Not Found '
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # pad to square
        h, w = original_image.shape[:2]
        if h != w:
            new_size = max(h, w)
            result_image = np.zeros((new_size, new_size, 3), dtype=original_image.dtype)
            result_image[0:h, 0:w, :] = original_image[0:h, 0:w, :]
        else:
            result_image = original_image

        transformed_image = image_tta(result_image)
        h, w = transformed_image.shape[:2]

        boxes, scores, labels = detect1Image(transformed_image, image_size, model, device, conf_thres, iou_thres)
        if len(boxes) > 0:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            for bi in range(boxes.shape[0]):
                boxes[bi, :] = box_reverse_tta(boxes[bi, :], h, w)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels

def predict_for_files(weights, folder, imagenames, image_size, is_TTA):
    if is_TTA:
        all_models_boxes = []
        all_models_scores = []
        all_models_labels = []

        for aug_ind in [0, 4]:
            print('aug_ind', aug_ind)
            all_boxes, all_scores, all_labels = predict_for_one_tta(weights, folder, imagenames, image_size, aug_ind)
            all_models_boxes.append(all_boxes)
            all_models_scores.append(all_scores)
            all_models_labels.append(all_labels)

        print('will merge #', len(all_models_boxes))
        return merge_boxes_from_models(all_models_boxes, all_models_scores,
                                       all_models_labels, len(imagenames), intersection_thr=0.4)
    else:
        return predict_for_one_tta(weights, folder, imagenames, image_size, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=-1, help='stage 1 or 2', required=True)
    opt = parser.parse_args()
    stage = opt.stage
    print(f'inference for stage {stage}')

    is_TTA = True
    folder = f'{CUR_PATH}data/test/'
    imagenames = os.listdir(folder)
    imagenames.sort() # for pretty output
    # For debugging
    # imagenames = imagenames[:10]
    image_ids = [x[:-4] for x in imagenames]
    print('images', len(image_ids))

    all_folds_boxes = []
    all_folds_scores = []
    all_folds_labels = []

    for fold in range(5):
        print('fold', fold)
        weights = f'{CUR_PATH}weights/stage{stage}_fold{fold}.pt'
        pred_boxes, pred_scores, pred_labels = predict_for_files(weights, folder, imagenames, 640, is_TTA)

        # Scale to initial size
        test_scaled_meta_df = pd.read_csv(f'{CUR_PATH}data/test.csv')
        for i, image in enumerate(image_ids):
            image_id = image_ids[i]
            image_width, image_height = test_scaled_meta_df.loc[test_scaled_meta_df[image_id_column] == image_id, ['width', 'height']].values[0]

            cur_boxes = pred_boxes[i]
            if len(cur_boxes) > 0:
                cur_boxes[:, [0, 2]] = (cur_boxes[:, [0, 2]] * image_width / 1024).astype(int)
                cur_boxes[:, [1, 3]] = (cur_boxes[:, [1, 3]] * image_height / 1024).astype(int)
            #else:
            #    cur_boxes = np.array((0, 4), dtype=int)

        all_folds_boxes.append(pred_boxes)
        all_folds_scores.append(pred_scores)
        all_folds_labels.append(pred_labels)

    result_boxes, result_scores, result_labels = merge_boxes_from_models(all_folds_boxes, all_folds_scores, all_folds_labels, len(image_ids), 0.4)
    write_submission(image_ids, result_boxes, result_scores, result_labels, 0, f'{CUR_PATH}yolo_stage{stage}_all_folds.csv')
