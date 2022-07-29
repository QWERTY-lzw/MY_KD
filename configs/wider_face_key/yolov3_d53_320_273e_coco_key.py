_base_ = '../yolo/yolov3_d53_320_273e_coco.py'

custom_imports = dict(
    imports=['mmdet_custom'],
    allow_failed_imports=False)

# model settings
model = dict(bbox_head=dict(num_classes=1))

# data settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypointAnnotations', with_bbox=True),
    dict(
        type='KExpand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='KMinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='KResize', img_scale=(320, 320), keep_ratio=True),
    dict(type='KRandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KeypointFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints'])
]


dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
data = dict(
    train=dict(
        type='KWIDERFaceDataset',
        ann_file=data_root + 'train_key.txt',
        img_prefix=data_root + 'WIDER_train/',
        min_size=17,
        ann_subdir='KAnnotations',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/'))

max_epochs = 273
num_last_epochs = 15
resume_from = None
interval = 20
checkpoint_config = dict(interval=interval)
evaluation = dict(
    metric='mAP', 
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    save_best='mAP')