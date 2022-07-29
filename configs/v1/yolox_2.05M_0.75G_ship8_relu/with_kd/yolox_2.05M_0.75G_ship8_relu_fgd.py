_base_ = '../yolox_2.05M_0.75G_ship8_relu.py'

custom_imports=dict(imports=['mmdet_custom.datasets','mmdet_custom.models','mmdet_custom.distillation'], allow_failed_imports=False) 

find_unused_parameters=True
temp=0.5
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.00010

distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = './work_dirs/yolox_tiny_300e_ship8_relu/latest.pth',
    init_student = False,
    yolox = True,
    distill_cfg = [
                    dict(student_module = 'neck.out_convs.0',
                         teacher_module = 'neck.out_convs.0',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 80,
                                       teacher_channels = 96,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.1',
                         teacher_module = 'neck.out_convs.1',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 80,
                                       teacher_channels = 96,
                                       temp = temp,
                                       alpha_fgd=1.5*alpha_fgd,
                                       beta_fgd=1.5*beta_fgd,
                                       gamma_fgd=1.5*gamma_fgd,
                                       lambda_fgd=1.5*lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.2',
                         teacher_module = 'neck.out_convs.2',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 80,
                                       teacher_channels = 96,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = './configs/v1/yolox_2.05M_0.75G_ship8_relu/yolox_2.05M_0.75G_ship8_relu.py'
teacher_cfg = './configs/v1/yolox_tiny_300e_ship8_relu.py'
