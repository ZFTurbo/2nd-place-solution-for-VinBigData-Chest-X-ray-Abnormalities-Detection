# coding: utf-8
__author__ = 'ZFTurbo: https://www.topcoder.com/members/ZFTurbo'

from a00_common_functions import read_single_image, save_in_file, CACHE_PATH
from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import (
    adjust_transform_for_image,
    apply_transform,
    resize_image,
)
from keras_retinanet.utils.transform import transform_aabb
from keras_retinanet.utils.anchors import (
    anchors_for_shape,
)
from keras_retinanet.utils.config import parse_anchor_parameters

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import random
import cv2
import time


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def get_class_index_arrays(classes_dict, image_data):
    classes = dict()
    classes['empty'] = set()
    for name in classes_dict:
        classes[classes_dict[name]] = set()

    for key in image_data:
        if len(image_data[key]) == 0:
            classes['empty'] |= set([key])
        for entry in image_data[key]:
            c = classes_dict[entry['class']]
            classes[c] |= set([key])

    for c in classes:
        classes[c] = list(classes[c])
        print('Class ID: {} Images: {}'.format(c, len(classes[c])))

    return classes


class CSVGeneratorCustom(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        base_dir=None,
        fraction=-1,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir
        self.fraction = fraction

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        print('Labels length: {}'.format(len(self.labels)))

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                reader = csv.reader(file, delimiter=',')
                try:
                    self.image_data = _read_annotations(reader, self.classes)
                except Exception as e1:
                    print('Try to skip first line with header!')
                    next(reader, None)
                    self.image_data = _read_annotations(reader, self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = sorted(list(self.image_data.keys()))

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.image_names)])
        self.image_id_to_id = dict([(k, i) for i, k in enumerate(self.image_names)])
        self.class_index_array = get_class_index_arrays(self.classes, self.image_data)
        self.check_labels_array = np.zeros(len(self.labels), dtype=np.int64)

        super(CSVGeneratorCustom, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_single_image(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path        = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            # Debug
            self.check_labels_array[self.name_to_label(annot['class'])] += 1

            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        # Debug - print classes distribution
        # print(list(self.check_labels_array), (self.check_labels_array == 0).astype(np.int32).sum())

        return self.compute_input_output(group)


    def preprocess_group_entry(self, image, annotations):

        """ Preprocess image and its annotations.
        """

        # Special case for very wide images try to scale it!
        # image, annotations['bboxes'] = fix_wide_images(image, annotations['bboxes'])

        if max(image.shape[0], image.shape[1]) > 2*self.image_max_side:
            image, image_scale = resize_image(image, min_side=2*self.image_min_side, max_side=2*self.image_max_side)
            annotations['bboxes'] *= image_scale

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        if 0:
            import cv2
            from a00_common_functions import show_image, show_resized_image, get_color
            image1 = image.copy()
            print(image1.shape, image1.min(), image1.max())
            print(annotations)
            for i, b in enumerate(annotations['bboxes']):
                b = b.astype(np.int32)
                color = get_color(int(annotations['labels'][i]))
                image1 = cv2.rectangle(image1, (b[0], b[1]), (b[2], b[3]), color, 2)
            if len(annotations['bboxes']) > 0:
                show_image(image1)

        # preprocess the image
        image = self.preprocess_image(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        debug = False
        # randomly transform both image and annotations
        if debug:
            import cv2
            from a00_common_functions import show_image, show_resized_image, get_color
            image1 = image.copy()
            print(image1.min(), image1.max())
            print(annotations)
            for i, b in enumerate(annotations['bboxes']):
                b = b.astype(np.int32)
                color = get_color(int(annotations['labels'][i]))
                image1 = cv2.rectangle(image1, (b[0], b[1]), (b[2], b[3]), color, 2)
            if len(annotations['bboxes']) > 0:
                show_resized_image(image1)

        if self.transform_generator:
            ann = dict()
            ann['image'] = image.copy()
            ann['labels'] = annotations['labels'].copy()
            ann['bboxes'] = list(annotations['bboxes'])
            try:
                augm = self.transform_generator(**ann)
                image = augm['image']
                annotations['bboxes'] = np.array(augm['bboxes'])
            except:
                print('Augm error')

        if debug:
            image1 = image.copy()
            print(annotations)
            for i, b in enumerate(annotations['bboxes']):
                b = b.astype(np.int32)
                color = get_color(int(annotations['labels'][i]))
                image1 = cv2.rectangle(image1, (b[0], b[1]), (b[2], b[3]), color, 2)
            if len(annotations['bboxes']) > 0:
                show_resized_image(image1)

        return image, annotations

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        random.seed(time.time())
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))
        elif self.group_method == 'random_classes':
            classes = list(range(self.num_classes())) + ['empty']
            self.groups = []
            while 1:
                if len(self.groups) > 1000000:
                    break
                self.groups.append([])
                for i in range(self.batch_size):
                    while 1:
                        random_class = random.choice(classes)
                        # print(random_class, len(self.class_index_array[random_class]))
                        if len(self.class_index_array[random_class]) > 0:
                            random_image = random.choice(self.class_index_array[random_class])
                            break
                    random_image_index = self.image_id_to_id[random_image]
                    self.groups[-1].append(random_image_index)
            # save_in_file((self.groups, self.image_id_to_id, self.id_to_image_id, self.image_names, self.image_data), CACHE_PATH + 'debug_generator.pklz')
            # exit()
            print('Grouped by random classes: {}'.format(len(self.groups)))
            return
        elif self.group_method == 'random_fraction':
            classes = list(range(self.num_classes())) + ['empty']
            print(classes)
            needed_fraction = self.fraction
            self.groups = []
            while 1:
                if len(self.groups) > 1000000:
                    break
                self.groups.append([])
                for i in range(self.batch_size):
                    if random.uniform(0, 1) < needed_fraction:
                        needed_class = 0
                    else:
                        needed_class = 'empty'
                    random_image = random.choice(self.class_index_array[needed_class])
                    random_image_index = self.image_id_to_id[random_image]
                    self.groups[-1].append(random_image_index)

            # save_in_file((self.groups, self.image_id_to_id, self.id_to_image_id, self.image_names, self.image_data), CACHE_PATH + 'debug_generator.pklz')
            # exit()
            print('Grouped by single non empty image per batch: {}'.format(len(self.groups)))
            print('Class fraction: {}'.format(needed_fraction))
            return

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    from albumentations import *
    from a00_common_functions import *

    config = dict()
    batch_size = 2
    input_size = 800
    multi_scale = True

    common_args = {
        'batch_size': batch_size,
        'image_min_side': input_size,
        'image_max_side': input_size,
    }

    fold = 0
    root_path = ROOT_PATH
    annotations = root_path + 'modified_data/centernet_data_full_png/fold_{}_train.csv'.format(fold)
    classes = root_path + 'modified_data/centernet_data_full_png/classes.txt'

    transform_generator = Compose([
        ShiftScaleRotate(p=0.9, shift_limit=0.1, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_REFLECT),
        RandomCropFromBorders(p=0.9, crop_value=0.1),
        OneOf([
            MedianBlur(p=1.0, blur_limit=7),
            Blur(p=1.0, blur_limit=7),
            GaussianBlur(p=1.0, blur_limit=7),
            GlassBlur(p=1.0, sigma=0.7, max_delta=2, iterations=2)
        ], p=0.2),
        RandomBrightnessContrast(p=0.9, brightness_limit=0.25, contrast_limit=0.25),
        OneOf([
            IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255), p=1.0),
            GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.5),
        IAAAffine(p=0.5, shear=(-10.0, 10.0)),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        HorizontalFlip(p=0.5),
    ], bbox_params={'format': 'pascal_voc',
                    'min_area': 1,
                    'min_visibility': 0.1,
                    'label_fields': ['labels']}, p=1.0)

    train_generator = CSVGeneratorCustom(
        annotations,
        classes,
        transform_generator=transform_generator,
        group_method='random',
        config=config,
        **common_args
    )
    for inputs, targets in train_generator:
        print(len(inputs), len(targets))
        # print(inputs[0].shape, inputs[1].shape, inputs[2].shape, inputs[3].shape, inputs[4].shape, inputs[5].shape)
        # print(targets[0], targets[1])