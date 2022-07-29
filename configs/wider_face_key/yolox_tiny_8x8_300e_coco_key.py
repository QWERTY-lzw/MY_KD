_base_ = '../yolox/yolox_tiny_8x8_300e_coco.py'

custom_imports = dict(
    imports=['mmdet_custom'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='KYOLOX',
    bbox_head=dict(type='KYOLOXHead', num_classes=1))

img_scale = (640, 640)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadKeypointAnnotations', with_bbox=True),
#     dict(
#         type='KExpand',
#         mean=img_norm_cfg['mean'],
#         to_rgb=img_norm_cfg['to_rgb'],
#         ratio_range=(1, 2)),
#     dict(
#         type='KMinIoURandomCrop',
#         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#         min_crop_size=0.3),
#     # dict(type='YOLOXHSVRandomAug'),
#     dict(type='KResize', img_scale=img_scale, keep_ratio=True),
#     dict(type='KRandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     # dict(type='KeypointFormatBundle'),
#     # dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints'])
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypointAnnotations', with_bbox=True),
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 2)),
    dict(
        type='KMinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='KResize', img_scale=img_scale, keep_ratio=True),
    dict(type='KRandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='KeypointFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints'])
]


# data settings
dataset_type = 'KWIDERFaceDataset'
data_root = 'data/WIDERFace/'
data = dict(
    train=dict(
        _delete_=True,
        type=dataset_type,
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

evaluation = dict(metric='mAP', save_best='mAP')