# coding: utf-8
__author__ = 'ZFTurbo: https://www.topcoder.com/members/ZFTurbo'

import argparse
import os
import sys
import warnings

sys.path.append(os.getcwd())

import keras
import keras.preprocessing.image
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.config import parse_anchor_parameters

from a00_common_functions import *
from keras_retinanet.callbacks.eval import Evaluate
# from net_v04_retinanet.a02_evaluate import Evaluate
from albumentations import *
from a01_adam_accumulate import AdamAccumulate, AccumOptimizer


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args, multi_gpu=0, freeze_backbone=False, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model


    # make prediction model
    prediction_model = retinanet_bbox(
        model=model,
        anchor_params=anchor_params,
        nms_threshold=args.nms_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    # compile model
    opt = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    if args.accum_iters > 1:
        print('Use AdamAccumulate: {} iters!'.format(args.accum_iters))
        # opt = AdamAccumulate(lr=1e-5, clipnorm=0.001, accum_iters=args.accum_iters)
        opt = AccumOptimizer(opt, args.accum_iters)


    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=opt
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.
    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        evaluation = Evaluate(
            validation_generator,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            tensorboard=tensorboard_callback
        )
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)


    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_last.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_{{mAP:.4f}}_{{epoch:02d}}.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
            save_best_only=False,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.9,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        min_delta = 0.0000001,
        cooldown = 0,
        min_lr   = 0
    ))

    callbacks.append(keras.callbacks.CSVLogger(
        args.snapshot_path + '{backbone}_fold_{fold}.csv'.format(backbone=os.path.basename(args.backbone), fold=args.fold), append=True,
    ))

    return callbacks


def create_generators(args, preprocess_image):
    global config

    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    from net_v04_retinanet.a01_csv_generator import CSVGeneratorCustom
    if 1:
        transform_generator = Compose([
            ShiftScaleRotate(p=0.9, shift_limit=0.1, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_REFLECT),
            IAAAffine(p=0.5, shear=(-10.0, 10.0)),
            # RandomCropFromBorders(p=0.9, crop_value=0.1),
            OneOf([
                MedianBlur(p=1.0, blur_limit=7),
                Blur(p=1.0, blur_limit=7),
                GaussianBlur(p=1.0, blur_limit=7),
                # GlassBlur(p=1.0, sigma=0.7, max_delta=2, iterations=2)
            ], p=0.2),
            RandomBrightnessContrast(p=0.9, brightness_limit=0.25, contrast_limit=0.25),
            OneOf([
                IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255), p=1.0),
                GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.5),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
            HorizontalFlip(p=0.5),
        ], bbox_params={'format': 'pascal_voc',
                        'min_area': 1,
                        'min_visibility': 0.1,
                        'label_fields': ['labels']}, p=1.0)
    else:
        transform_generator = Compose([
            HorizontalFlip(p=0.5),
        ], bbox_params={'format': 'pascal_voc',
                        'min_area': 1,
                        'min_visibility': 0.1,
                        'label_fields': ['labels']}, p=1.0)

    train_generator = CSVGeneratorCustom(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        group_method='random_classes',
        config=config,
        **common_args
    )

    if args.val_annotations:
        validation_generator = CSVGeneratorCustom(
            args.val_annotations,
            args.classes,
            group_method='',
            config=config,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--fixed-labels', help='Use the exact specified labels.', default=False)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--fold',            help='Fold number.', type=int, default=1)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=3000)
    parser.add_argument('--lr',              help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--nms_threshold',   help='NMS threshold.', type=float, default=0.5)
    parser.add_argument('--max_detections',  help='Max detections.', type=int, default=100)
    parser.add_argument('--score_threshold', help='Score threshold.', type=float, default=0.01)
    parser.add_argument('--iou_threshold',   help='IOU threshold.', type=float, default=0.5)
    parser.add_argument('--accum_iters',     help='Accum iters. If more than 1 used AdamAccum optimizer', type=int, default=1)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return check_args(parser.parse_args(args))


def main(args=None):
    global config
    from keras import backend as K

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print('Arguments: {}'.format(args))

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if 'anchor_parameters' in config:
            anchor_params = parse_anchor_parameters(config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            args=args,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            config=config
        )

    # print model summary
    print(model.summary())

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    if args.lr > 0.0:
        K.set_value(model.optimizer.lr, args.lr)
        print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    init_epoch = 0
    try:
        if args.snapshot:
            init_epoch = int(args.snapshot.split("_")[-2])
    except:
        pass
    # init_epoch = 6
    print('Init epoch: {}'.format(init_epoch))

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=init_epoch,
    )


if __name__ == '__main__':
    random.seed(time.time())
    config = dict()

    gpu_id = 0
    fold = 0
    print('GPU: {} Fold: {}'.format(gpu_id, fold))
    root_path = ROOT_PATH
    models_path = MODELS_PATH + 'retinanet_resnet101_sqr/'
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    params = [
        # '--snapshot', models_path + 'resnet50_fold_0_226_0.5380.h5',
        # '--imagenet-weights',
        '--weights', MODELS_PATH + 'retinanet_resnet101_500_classes_0.4986.h5',
        # '--weights', MODELS_PATH + 'retinanet_resnet101/best/resnet101_fold_{}_0.3465_64.h5'.format(fold),
        # '--weights', models_path + 'resnet152_fold_{}_0.3396_57.h5'.format(fold),
        '--gpu', str(gpu_id),
        '--steps', '5000',
        '--epochs', '1000',
        '--snapshot-path', models_path,
        '--lr', '1e-5',
        '--nms_threshold', '0.3',
        '--max_detections', '200',
        '--score_threshold', '0.01',
        '--iou_threshold', '0.4',
        '--fold', str(fold),
        '--accum_iters', '10',
        # '--multi-gpu', '2',
        # '--multi-gpu-force',
        '--backbone', 'resnet101',
        # '--freeze-backbone',
        '--batch-size', '1',
        '--image-min-side', '1024',
        '--image-max-side', '1024',
        'csv',
        root_path + 'modified_data/retinanet_train_sqr_data/fold_{}_train.csv'.format(fold),
        root_path + 'modified_data/retinanet_train_sqr_data/classes.txt',
        '--val-annotations', root_path + 'modified_data/retinanet_train_sqr_data/fold_{}_valid.csv'.format(fold),
    ]
    main(params)

