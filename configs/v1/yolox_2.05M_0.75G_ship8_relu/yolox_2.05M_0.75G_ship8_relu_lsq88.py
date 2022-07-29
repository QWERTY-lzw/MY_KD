_base_ = './yolox_2.05M_0.75G_ship8_relu.py'

optimizer = dict(lr=0.001)

quant = True
qconv_type = 'QConv2d'
qact_type = 'QReLU'
w_bit = 8
channel_wise = True
a_bit = 8
quant_cfg = {
    'quant': quant, 'w_bit': w_bit, 'a_bit': a_bit, 'channel_wise': channel_wise
}

pretrained = './work_dirs/yolox_2.05M_0.75G_ship8_relu/latest.pth'
model = dict(
    quant_cfg=quant_cfg,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=pretrained,
        map_location='cpu'),
    backbone=dict(
        type='QuantSearchableCSPDarknet',
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit)),
    neck=dict(
        type='QuantSearchableYOLOXPAFPN',
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit)),
    bbox_head=dict(
        conv_cfg=dict(type=qconv_type, w_bit=w_bit, channel_wise=channel_wise),
        act_cfg=dict(type=qact_type, a_bit=a_bit))
)
