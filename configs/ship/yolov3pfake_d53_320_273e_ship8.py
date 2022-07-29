_base_ = '../yolo/yolov3_d53_320_273e_coco.py'

custom_imports=dict(imports='mmdet_custom.datasets', allow_failed_imports=False) 

# model settings
model = dict(
    type='YOLOV3',
    bbox_head=dict(
        type='YOLOV3PHead',
        num_classes=8,
        loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=100.0),
        loss_offset=dict(type='L1Loss', loss_weight=100.0)
    ),
    test_cfg=dict(topk=100, local_maximum_kernel=7),
)
# dataset settings
dataset_type = 'SHIPDataset8'
data_root = 'data/ship-comp-committee-new/'

data = dict(
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root),
    test=dict(
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