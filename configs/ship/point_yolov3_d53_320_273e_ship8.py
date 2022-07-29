_base_ = '../yolo/yolov3_d53_320_273e_coco.py'

custom_imports=dict(imports='mmdet_custom.datasets', allow_failed_imports=False) 

img_scale = (320, 256)
# model settings
model = dict(
    type='Point_YOLOV3',
    bbox_head=dict(
        num_classes=5,
    ),
    point_head=dict(
        type='PointHead',
        in_channel=128,
        feat_channel=128,
        num_classes=3,
        test_cfg=dict(topk=100, local_maximum_kernel=3),
    ),
)

# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'SHIPDataset8'
data_root = 'data/ship-comp-committee-new/'

data = dict(
    persistent_workers=True,
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root),
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root),
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root))

max_epochs = 273
num_last_epochs = 5
resume_from = None
interval = 50
checkpoint_config = dict(interval=interval)
evaluation = dict(
    metric='mAP', 
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    save_best='mAP')

custom_hooks=[]