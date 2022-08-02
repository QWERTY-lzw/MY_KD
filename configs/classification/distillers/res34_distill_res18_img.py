_base_ = [
    '../resnet/resnet18_b32x8_imagenet.py'
]
# model settings
#custom_imports=dict(imports=['mmcls_custom.models','mmcls_custom.distillation'], allow_failed_imports=False) 
find_unused_parameters=True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth',
    distill_cfg = [ 
                    dict(methods=[dict(type='SimKDLoss',
                                       name='loss_simkd',
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha_simkd=7e-5,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/classification/resnet/resnet18_b32x8_imagenet.py'
teacher_cfg = 'configs/classification/resnet/resnet34_b32x8_imagenet.py'
dataset_type = 'ImageNet100'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet100/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet100/val',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet100/val',
        ann_file=None,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))