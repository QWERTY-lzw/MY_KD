_base_ = '../ship8/relu/yolox_tiny_0.33w_300e_ship8_relu.py'

optimizer = dict(lr=0.001)

quant = True
qconv_type = 'QConv2d'
qact_type = 'QReLU'
w_bit = 4
channel_wise = True
a_bit = 4

pretrained = './work_dirs/yolox_tiny_0.33w_300e_ship8_relu/latest.pth'
model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint=pretrained,
        map_location='cpu'),
    backbone=dict(
        type='QuantCSPDarknet',
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit)),
    neck=dict(
        type='QuantYOLOXPAFPN',
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit)),
    bbox_head=dict(
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit))
)
