_base_ = '../yolox/yolox_tiny_8x8_300e_coco.py'

# model settings
model = dict(bbox_head=dict(num_classes=1))

# data settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.txt',
            img_prefix=data_root + 'WIDER_train/',
            min_size=17)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/'))

evaluation = dict(metric='mAP', save_best='mAP')