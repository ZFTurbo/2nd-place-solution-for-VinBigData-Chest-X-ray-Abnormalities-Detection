# coding: utf-8
__author__ = 'ZFTurbo: https://www.topcoder.com/members/ZFTurbo'

if __name__ == '__main__':
    import os
    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import argparse
import sys

sys.path.append(os.getcwd())

from a00_common_functions import *
from a01_adam_accumulate import AdamAccumulate, AccumOptimizer
from keras.optimizers import Adam


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')

    return parser.parse_args(args)


def main(args=None):
    from keras_retinanet.utils.config import parse_anchor_parameters
    from keras.utils import custom_object_scope

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    anchor_params = None
    if 0:
        config = dict()
        config['anchor_parameters'] = dict()
        config['anchor_parameters']['sizes'] = '16 32 64 128 256 512'
        config['anchor_parameters']['strides'] = '8 16 32 64 128'
        config['anchor_parameters']['ratios'] = '0.1 0.5 1 2 4 8'
        config['anchor_parameters']['scales'] = '1 1.25 1.5 1.75'
        anchor_params = parse_anchor_parameters(config)

    # load and convert model
    with custom_object_scope({'AdamAccumulate': AdamAccumulate, 'AccumOptimizer': Adam}):
        model = models.load_model(args.model_in,  backbone_name=args.backbone)
        model = models.convert_model(
            model,
            nms=args.nms,
            class_specific_filter=args.class_specific_filter,
            max_detections=500,
            nms_threshold=0.3,
            score_threshold=0.01,
            anchor_params=anchor_params
        )

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    model_type = 'resnet101'
    model_list = [
        MODELS_PATH + 'retinanet_resnet101_sqr_removed_rads/best/resnet101_fold_0_0.1817_05.h5',
        MODELS_PATH + 'retinanet_resnet101_sqr_removed_rads/best/resnet101_fold_1_0.2072_19.h5',
        MODELS_PATH + 'retinanet_resnet101_sqr_removed_rads/best/resnet101_fold_2_0.1938_03.h5',
        MODELS_PATH + 'retinanet_resnet101_sqr_removed_rads/best/resnet101_fold_3_0.1884_07.h5',
        MODELS_PATH + 'retinanet_resnet101_sqr_removed_rads/best/resnet101_fold_4_0.2227_05.h5',
    ]
    for m in model_list:
        params = [
            m,
            m[:-3] + '_iou_0.3_converted.h5',
            '--backbone', model_type
        ]
        main(params)
