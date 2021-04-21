# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
import cv2


def show_image_debug(id_to_labels, draw, boxes, scores, labels):
    from keras_retinanet.utils.visualization import draw_box, draw_caption
    from keras_retinanet.utils.colors import label_color

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.4:
            break

        color = (0, 255, 0)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(id_to_labels[label], score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    show_resized_image(draw)


def get_retinanet_predictions_for_files(model_path, files, out_dir, backbone, min_width, max_width, lvl_labels):
    from keras_retinanet.utils.image import preprocess_image, resize_image
    from keras_retinanet import models

    show_debug_images = False
    show_mirror_predictions = False

    model = models.load_model(model_path, backbone_name=backbone)
    print('Proc {} files...'.format(len(files)))
    for f in files:
        id = os.path.basename(f)[:-4]

        cache_path = out_dir + id + '.pkl'
        if os.path.isfile(cache_path):
           continue

        # try:
        image = read_single_image(f)

        if show_debug_images:
            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=min_width, max_side=max_width)

        # Add mirror
        image = np.stack((image, image[:, ::-1, :]), axis=0)

        # process image
        start = time.time()
        print('ID: {} Image shape: {} Scale: {}'.format(id, image.shape, scale))
        boxes, scores, labels = model.predict_on_batch(image)
        print('Detections shape: {} {} {}'.format(boxes.shape, scores.shape, labels.shape))
        print("Processing time: {:.2f} sec".format(time.time() - start))

        if show_debug_images:
            if show_mirror_predictions:
                draw = draw[:, ::-1, :]
            boxes_init = boxes.copy()
            boxes_init /= scale

        boxes[:, :, 0] /= image.shape[2]
        boxes[:, :, 2] /= image.shape[2]
        boxes[:, :, 1] /= image.shape[1]
        boxes[:, :, 3] /= image.shape[1]

        if show_debug_images:
            if show_mirror_predictions:
                show_image_debug(lvl_labels, draw.astype(np.uint8), boxes_init[1:], scores[1:], labels[1:])
            else:
                show_image_debug(lvl_labels, draw.astype(np.uint8), boxes_init[:1], scores[:1], labels[:1])

        save_in_file_fast((boxes, scores, labels), cache_path)


def get_retinanet_preds_for_tst(files, model_path, backbone, reverse, min_width, max_width, out_path_prefix, labels):
    if reverse is True:
        files = files[::-1]
    if not os.path.isdir(out_path_prefix + '_test/'):
        os.mkdir(out_path_prefix + '_test/')
    get_retinanet_predictions_for_files(model_path, files, out_path_prefix + '_test/', backbone, min_width, max_width, labels)


def get_retinanet_preds_for_valid(files, model_path, backbone, reverse, min_width, max_width, out_path_prefix, labels):
    if reverse is True:
        files = files[::-1]
    if not os.path.isdir(out_path_prefix + '_valid/'):
        os.mkdir(out_path_prefix + '_valid/')
    get_retinanet_predictions_for_files(model_path, files, out_path_prefix + '_valid/', backbone, min_width, max_width, labels)


if __name__ == '__main__':

    mp = MODELS_PATH + 'retinanet_resnet101_sqr/'
    model_list = [
        mp + 'resnet101_fold_0_0.3573_26_iou_0.3_converted.h5',
        mp + 'resnet101_fold_1_0.3481_35_iou_0.3_converted.h5',
        mp + 'resnet101_fold_2_0.3804_24_iou_0.3_converted.h5',
        mp + 'resnet101_fold_3_0.3584_24_iou_0.3_converted.h5',
        mp + 'resnet101_fold_4_0.3514_12_iou_0.3_converted.h5',
    ]

    calc_fold = [0, 1, 2, 3, 4]

    for fold in range(len(model_list)):
        if fold not in calc_fold:
            continue
        print('Go fold: {}'.format(fold))
        reverse = False
        model_path = model_list[fold]
        min_width = 1024
        max_width = 1024
        backbone = 'resnet101'
        out_path_prefix = OUTPUT_PATH + os.path.basename(model_path)[:-13]
        labels = get_classes_array()
        # valid_files = sorted(pd.read_csv(OUTPUT_PATH + 'retinanet_div_2/fold_{}_valid.csv'.format(fold))['id'].unique())
        # get_retinanet_preds_for_valid(valid_files, model_path, backbone, reverse, min_width, max_width, out_path_prefix, labels)
        test_files = sorted(glob.glob(INPUT_PATH + 'test_png_div_2/*.png'))
        get_retinanet_preds_for_tst(test_files, model_path, backbone, reverse, min_width, max_width, out_path_prefix, labels)

