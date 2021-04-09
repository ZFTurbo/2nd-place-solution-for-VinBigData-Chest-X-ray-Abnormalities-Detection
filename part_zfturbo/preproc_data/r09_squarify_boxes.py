# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from preproc_data.r05_merge_boxes_v1 import create_split_for_centernet, create_test_file_centernet


def draw_box(image_id, x1, y1 ,x2, y2):
    image = read_single_image(INPUT_PATH + 'train_png_div_4/{}.png'.format(image_id))
    print(image.shape, image.min(), image.max())
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    show_resized_image(image)


def squarify_boxes(split_file, div_val):
    draw = False
    sizes = get_train_test_image_sizes()
    out = open(split_file[:-4] + '_sqr.csv', 'w')
    out.write('id,x1,y1,x2,y2,class,score\n')
    data = pd.read_csv(split_file)
    ratios = []
    for index, row in data.iterrows():
        image_id, cls, score = row['id'], row['class'], row['score']
        if str(cls) == 'nan':
            out.write("{},,,,,,\n".format(image_id))
            continue
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        delta_x = x2 - x1
        delta_y = y2 - y1
        if delta_x > delta_y:
            ar = delta_x / delta_y
        else:
            ar = delta_y / delta_x
        ratios.append(ar)
        if ar > 2:
            print('Initial AR: {:.2f}'.format(ar))
            if draw:
                draw_box(image_id, x1, y1, x2, y2)
            if delta_x > delta_y:
                if ar > 3:
                    # Чутка уменьшаем Delta_x
                    move_x = int(delta_x * 0.05)
                    x1 = int(x1 + move_x)
                    x2 = int(x2 - move_x)
                # Потом двигаем y относительно середины
                t = ((y2 - y1) / 2) - ((x2 - x1) / 4)
                y2 -= int(t)
                y1 += int(t)
                ar = (x2 - x1) / (y2 - y1)
            else:
                if ar > 3:
                    # Чутка уменьшаем Delta_y
                    move_y = int(delta_y * 0.05)
                    y1 = int(y1 + move_y)
                    y2 = int(y2 - move_y)
                # Потом двигаем x относительно середины
                t = ((x2 - x1) / 2) - ((y2 - y1) / 4)
                x2 -= int(t)
                x1 += int(t)
                ar = (y2 - y1) / (x2 - x1)
            if draw:
                draw_box(image_id, x1, y1, x2, y2)
            print('New AR: {:.2f}'.format(ar))

        if x1 < 0:
            print('Zero x1! {}'.format(x1))
            # draw_box(image_id, x1, y1, x2, y2)
            x1 = 0
        if y1 < 0:
            print('Zero y1! {}'.format(y1))
            # draw_box(image_id, x1, y1, x2, y2)
            y1 = 0
        if x2 >= sizes[image_id][1] / div_val:
            print('Large x2! {}'.format(x2))
            # draw_box(image_id, x1, y1, x2, y2)
            x2 = sizes[image_id][1] // div_val - 1
        if y2 >= sizes[image_id][0] / div_val:
            print('Large y2! {}'.format(y2))
            # draw_box(image_id, x1, y1, x2, y2)
            y2 = sizes[image_id][0] // div_val - 1
        out.write("{},{},{},{},{},{},{}\n".format(image_id, x1, y1, x2, y2, cls, score))

    ratios = np.array(ratios)
    print('AR Min: {:.2f} AR Max: {:.2f} AR Mean: {:.2f}'.format(ratios.min(), ratios.max(), ratios.mean()))
    out.close()


if __name__ == '__main__':
    initial_file = OUTPUT_PATH + 'boxes_description_iou_0.4_div_2.csv'
    squarify_boxes(initial_file, div_val=2)

    out_folder = OUTPUT_PATH + 'retinanet_train_sqr_data/'
    training_img_directory = INPUT_PATH + 'train_png_div_2/'
    testing_img_directory = INPUT_PATH + 'test_png_div_2/'

    create_split_for_centernet(initial_file[:-4] + '_sqr.csv', training_img_directory, out_folder)
    create_test_file_centernet(testing_img_directory, out_folder)


"""
Class: Aortic enlargement Number: 3204 AR Min: 1.00 AR Max: 3.88 AR Mean: 1.17
Class: Atelectasis Number: 226 AR Min: 1.00 AR Max: 6.07 AR Mean: 1.67
Class: Calcification Number: 726 AR Min: 1.00 AR Max: 5.40 AR Mean: 1.56
Class: Cardiomegaly Number: 2335 AR Min: 1.00 AR Max: 8.17 AR Mean: 2.81
Class: Consolidation Number: 428 AR Min: 1.00 AR Max: 3.05 AR Mean: 1.41
Class: ILD Number: 690 AR Min: 1.00 AR Max: 4.85 AR Mean: 1.81
Class: Infiltration Number: 914 AR Min: 1.00 AR Max: 3.81 AR Mean: 1.45
Class: Lung Opacity Number: 1939 AR Min: 1.00 AR Max: 6.21 AR Mean: 1.55
Class: Nodule/Mass Number: 1825 AR Min: 1.00 AR Max: 4.81 AR Mean: 1.26
Class: Other lesion Number: 1776 AR Min: 1.00 AR Max: 10.29 AR Mean: 2.03
Class: Pleural effusion Number: 1606 AR Min: 1.00 AR Max: 33.00 AR Mean: 1.67
Class: Pleural thickening Number: 3797 AR Min: 1.00 AR Max: 9.78 AR Mean: 2.37
Class: Pneumothorax Number: 128 AR Min: 1.01 AR Max: 5.86 AR Mean: 1.86
Class: Pulmonary fibrosis Number: 3125 AR Min: 1.00 AR Max: 10.28 AR Mean: 1.74
"""