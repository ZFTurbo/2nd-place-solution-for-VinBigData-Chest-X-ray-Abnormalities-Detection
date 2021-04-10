import os
import numpy as np
import csv
import pandas as pd
import argparse
from ensemble_boxes import *


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
            # print(pred_variants[vi])
            if len(pred_variants[vi]) > 0:
                cur_pred_labels, cur_pred_scores, cur_pred_boxes = list(zip(*pred_variants[vi]))
            else:
                cur_pred_labels = []
                cur_pred_scores = []
                cur_pred_boxes = []
            # print(cur_pred_boxes)
            cur_pred_boxes = np.array(cur_pred_boxes, copy=False)
            cur_pred_scores = np.array(cur_pred_scores, copy=False)
            cur_pred_labels = np.array(cur_pred_labels, copy=False)

            # WBF expects the coordinates in 0-1 range.
            cur_pred_boxes = cur_pred_boxes / max_value

            pred_boxes.append(cur_pred_boxes)
            pred_scores.append(cur_pred_scores)
            pred_labels.append(cur_pred_labels)

        """
        print(pred_boxes)
        print('-' * 20)
        print(pred_scores)
        print('-' * 20)
        print(pred_labels)
        """

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
        # pred_boxes = make_boxes_int(pred_boxes)
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
            pred_strings.append(
                "{0} {1:.4f} {2} {3} {4} {5}".format(int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
    )
    parser.add_argument(
        '--resolution', default=1024
    )

    hyperparams = parser.parse_args()

    model_name = hyperparams.model_name
    resolution = hyperparams.resolution

    test_files = ['subs/{}_{}/{}'.format(model_name, resolution, x) for x in os.listdir('subs/{}_{}'.format(model_name, resolution))]

    print(test_files)

    print('# models', len(test_files))

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
                    if label < 14:
                        locations.append((label, score, location))
                if filename in pred_locations:
                    pred_locations[filename].append(locations)
                else:
                    pred_locations[filename] = [locations]

    test_filenames = [*pred_locations]
    test_filenames.sort()

    result_boxes, result_scores, result_labels = merge_boxes_from_models(0.4)
    write_submission(test_filenames, result_boxes, result_scores, result_labels, 0, 'subs/{}_{}/{}_5folds_{}_no_postprocess.csv'.format(model_name, resolution, model_name, resolution))
