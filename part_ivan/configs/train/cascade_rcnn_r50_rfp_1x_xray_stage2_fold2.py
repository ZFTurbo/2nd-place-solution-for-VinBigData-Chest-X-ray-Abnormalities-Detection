_base_ = [
    '../../mmdetection/configs/detectors/cascade_rcnn_r50_rfp_1x_coco.py'
]

#fp16 = dict(loss_scale=512.)

model = dict(
        roi_head=dict(
            bbox_head=[
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=14,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=14,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=14,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

dataset_type = 'COCODataset'

classes = ('Aortic_enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
           'Lung_Opacity', 'Nodule/Mass',
           'Other_lesion', 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax', 'Pulmonary_fibrosis')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.1,
        rotate_limit=20,
        scale_limit=0.2,
       # border_mode=cv2.BORDER_REFLECT,
        p=0.9
    ),
    dict(
        type='IAAAffine',
        shear=(-10.0, 10.0),
        p=0.5
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", p=1.0, blur_limit=7),
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.2,
    ),
    dict(type="RandomBrightnessContrast", p=0.9, brightness_limit=0.25, contrast_limit=0.25),
    dict(
        type='OneOf',
        transforms=[
            dict(type='IAAAdditiveGaussianNoise', scale=(0.01 * 255, 0.05 * 255), p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0)
        ]
    ),
    dict(type='HorizontalFlip', p=0.5)
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=['gt_labels'], filter_lost_elements=True, min_visibility=0.1, min_area=1),
        ),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
    train=dict(
        img_prefix='data',
        classes=classes,
        ann_file='data/folds/coco_train_fold_2_stage2.json',
        pipeline = train_pipeline,
        filter_empty_gt=False),
    val=dict(
        img_prefix='data',
        classes=classes,
        ann_file='data/folds/coco_valid_fold_2_stage2.json',
        pipeline = test_pipeline,
        filter_empty_gt=False))


log_config = dict(  # config to register logger hook
    interval=100,  # Interval to print the log
    hooks=[
        dict(type='TensorboardLoggerHook'), # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.0)

total_epochs = 10
runner = dict(max_epochs = 10)

load_from = 'weights/cascade_r50_augs_with_empty/fold2.pth'
work_dir = 'work_dirs/cascade_r50_augs_rare_with_empty_fold2'

