_base_ = '../yolo/yolov3_d53_320_273e_coco.py'

custom_imports=dict(imports='mmdet_custom.datasets', allow_failed_imports=False) 

# model settings
model = dict(bbox_head=dict(num_classes=8))
# dataset settings
dataset_type = 'SHIPDataset8'
data_root = 'data/ship-comp-committee-new/'

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
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
data = dict(
    persistent_workers=True,
    train=dict(
        pipeline=train_pipeline,
        # pipeline=test_pipeline,
        type=dataset_type,
        # ann_file=data_root + 'train.txt',
        ann_file=data_root + 'val.txt',
        img_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root))

