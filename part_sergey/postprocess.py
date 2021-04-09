import numpy as np
import csv
import pandas as pd
import argparse

image_id_column = 'image_id'
is_healthy_column = 'is_healthy'

pred_2class = pd.read_csv("../MegaMix/341_healthy.csv")

NORMAL = "14 1 0 0 1 1"
#low_threshold = 0.0
high_threshold = 1

image_id_column = 'image_id'

def load_predictions(filename):
    pred_locations = {}
    # load table
    with open(filename, mode='r') as infile:
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
                locations.append((label, score, location))
            pred_locations[filename] = locations
    return pred_locations

def format_prediction_string(boxes, scores, labels):
    pred_strings = []
    if len(boxes) > 0 :
        for j in zip(labels, scores, boxes):
            pred_strings.append("{0} {1} {2} {3} {4} {5}".format(int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))
    else:
        pred_strings.append("14 1 0 0 1 1")

    return " ".join(pred_strings)

def write_submission(test_images_ids, result_boxes, result_scores, result_labels, submission_file_name):
    results = []
    for i, image in enumerate(test_images_ids):
        image_id = test_images_ids[i]
        cur_boxes = np.array(result_boxes[i])
        cur_scores = np.array(result_scores[i])
        cur_labels = np.array(result_labels[i])

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(cur_boxes, cur_scores, cur_labels)
        }
        results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv(submission_file_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='', help='file to preprocess', required=True)
    opt = parser.parse_args()

    test_result = opt.f
    output_file = test_result[:-4] + "_healthy_341_postprocessed.csv"
    print('test_result:', test_result)

    test_locations = load_predictions(test_result)

    pred_det_df = pd.read_csv(test_result)
    n_normal_before = len(pred_det_df.query("PredictionString == @NORMAL"))
    merged_df = pd.merge(pred_det_df, pred_2class, on="image_id", how="left").reset_index()

    pred_boxes = []
    pred_labels = []
    pred_scores = []

    for ind in merged_df.index:
        p0 = merged_df.loc[ind, is_healthy_column]
        test_filename = merged_df.loc[ind, image_id_column]

        cur_image_pred_boxes = []
        cur_image_pred_labels = []
        cur_image_pred_scores = []
        has_healthy = False
        for label, score, location in test_locations[test_filename]:
            if label == 14:
                has_healthy = True
                cur_image_pred_boxes.append(location)
                cur_image_pred_labels.append(label)
                cur_image_pred_scores.append(p0)
            else:
                if p0 < high_threshold:
                    cur_image_pred_boxes.append(location)
                    cur_image_pred_labels.append(label)
                    cur_image_pred_scores.append(score)
        if not has_healthy:
            cur_image_pred_boxes.append([0, 0, 1, 1])
            cur_image_pred_labels.append(14)
            cur_image_pred_scores.append(p0)

        pred_boxes.append(cur_image_pred_boxes)
        pred_labels.append(cur_image_pred_labels)
        pred_scores.append(cur_image_pred_scores)

    n_normal_after = len(merged_df.query("PredictionString == @NORMAL"))
    print(f"n_normal: {n_normal_before} -> {n_normal_after} with threshold {high_threshold}")

    write_submission(merged_df[image_id_column].values, pred_boxes, pred_scores, pred_labels, output_file)

