_base_ = [
    './autoslim_yolox_tinyx2_300e_ship8.py',
]

algorithm = dict(
    distiller=None,
    retraining=True,
    bn_training_mode=False,
)

find_unused_parameters = True

max_epochs = 300
num_last_epochs = 5
resume_from = None
interval = 50
custom_hooks = [
    # dict(
    #     type='YOLOXModeSwitchHook',
    #     num_last_epochs=num_last_epochs,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    # dict(
    #     type='ExpMomentumEMAHook',
    #     resume_from=resume_from,
    #     momentum=0.0001,
    #     priority=49)
]