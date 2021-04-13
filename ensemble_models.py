import numpy as np
import csv
import pandas as pd
from ensemble_boxes import *

test_files = [
    './subm_folder/yolo_stage2_all_folds.csv',
    './subm_folder/ensemble_retinanet_resnet101_removed_rad.csv',
    './subm_folder/cascade_r50_augs_rare_with_empty_5folds_1024_postprocess.csv',
    './subm_folder/ensemble_retinanet_resnet101_sqr.csv',
    './subm_folder/ensemble_yolo_final.csv',
    './subm_folder/cascade_r50_augs_with_empty_5folds_1024_postprocess.csv'
]
print('# models', len(test_files))

blend_weights = None
print(test_files)

pred_locations = {}

for model_index in range(len(test_files)):
    # load table
    with open(test_files[model_index], mode='r') as infile:
        # open reader
        reader = csv.reader(infile)
        # skip header
        next(reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows[0]
            # print(rows[0])
            parts = rows[1].split()
            # print(parts)
            assert len(parts) % 6 == 0
            locations = []
            for ind in range(len(parts) // 6):
                label = int(parts[ind * 6])
                score = float(parts[ind * 6 + 1])
                location = int(float(parts[ind * 6 + 2])), int(float(parts[ind * 6 + 3])), \
                           int(float(parts[ind * 6 + 4])), int(float(parts[ind * 6 + 5]))
                if score > 0 and label < 14:
                    locations.append((label, score, location))
            if filename in pred_locations:
                pred_locations[filename].append(locations)
            else:
                pred_locations[filename] = [locations]

test_filenames = [*pred_locations]
test_filenames.sort()

def merge_boxes_from_models(intersection_thr):
    n_images = len(test_filenames)
    n_variants = len(test_files)

    result_boxes = []
    result_scores = []
    result_labels = []
    for ii in range(n_images):
        image_id = test_filenames[ii]
        pred_variants = pred_locations[image_id]
        assert len(pred_variants) == n_variants

        pred_boxes = []
        pred_scores = []
        pred_labels = []
        max_value = 10000
        for vi in range(n_variants):
            if len(pred_variants[vi]) > 0:
                cur_pred_labels, cur_pred_scores, cur_pred_boxes = list(zip(*pred_variants[vi]))
            else:
                cur_pred_labels = []
                cur_pred_scores = []
                cur_pred_boxes = []
            cur_pred_boxes = np.array(cur_pred_boxes, copy=False)
            cur_pred_scores = np.array(cur_pred_scores, copy=False)
            cur_pred_labels = np.array(cur_pred_labels, copy=False)

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
            weights=blend_weights,
            iou_thr=intersection_thr,
            skip_box_thr=0.,
            allows_overflow=True
        )
        pred_boxes = np.round(pred_boxes * max_value).astype(int)

        assert len(pred_boxes) == len(pred_scores)
        assert len(pred_boxes) == len(pred_scores)
        assert len(pred_boxes) == len(pred_labels)
        result_boxes.append(pred_boxes)
        result_scores.append(pred_scores)
        result_labels.append(pred_labels)

    return result_boxes, result_scores, result_labels


def format_prediction_string(boxes, scores, labels):
    pred_strings = []
    if len(boxes) > 0:
        for j in zip(labels, scores, boxes):
            pred_strings.append("{0} {1} {2} {3} {4} {5}".format(int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))
    else:
        pass
        #pred_strings.append("14 1 0 0 1 1")

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


def concat_lists(class_boxes1, class_scores1, class_labels1,  class_boxes2, class_scores2, class_labels2):
    n_images = len(test_filenames)
    for ii in range(n_images):
        class_boxes1[ii] = np.concatenate((class_boxes1[ii], class_boxes2[ii]), axis=0)
        class_scores1[ii] = np.concatenate((class_scores1[ii], class_scores2[ii]), axis=0)
        class_labels1[ii] = np.concatenate((class_labels1[ii], class_labels2[ii]), axis=0)


result_boxes, result_scores, result_labels = merge_boxes_from_models(0.4)

# This classifier is taken from the public notebook.
# https://www.kaggle.com/awsaf49/vinbigdata-2class-prediction
bin_class_df = pd.read_csv('./subm_folder/2-cls test pred.csv')

class14_boxes = []
class14_scores = []
class14_labels = []
for test_filename in test_filenames:
    p0 = 1 - bin_class_df.loc[bin_class_df['image_id'] == test_filename, 'target'].values[0]

    cur_image_pred_boxes = []
    cur_image_pred_labels = []
    cur_image_pred_scores = []
    cur_image_pred_boxes.append([0, 0, 1, 1])
    cur_image_pred_labels.append(14)
    cur_image_pred_scores.append(p0)
    class14_boxes.append(cur_image_pred_boxes)
    class14_scores.append(cur_image_pred_scores)
    class14_labels.append(cur_image_pred_labels)

concat_lists(result_boxes, result_scores, result_labels, class14_boxes, class14_scores, class14_labels)

write_submission(test_filenames, result_boxes, result_scores, result_labels, 0, 'submission_ensemble.csv')
