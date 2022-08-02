_base_ = './retinanet_r50_fpn_2x_coco.py'
# model settings
find_unused_parameters=True
alpha_simkd=2e-5
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth',
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='SimKDLoss',
                                       name='loss_mgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_simkd=alpha_simkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='SimKDLoss',
                                       name='loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_simkd=alpha_simkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='SimKDLoss',
                                       name='loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_simkd=alpha_simkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='SimKDLoss',
                                       name='loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_simkd=alpha_simkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='SimKDLoss',
                                       name='loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_simkd=alpha_simkd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
'''
fp16 = dict(loss_scale=512.)
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/'))
'''