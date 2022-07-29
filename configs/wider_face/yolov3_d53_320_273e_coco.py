_base_ = '../yolo/yolov3_d53_320_273e_coco.py'

# model settings
model = dict(bbox_head=dict(num_classes=1))

# data settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root + 'WIDER_train/',
        min_size=17),
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