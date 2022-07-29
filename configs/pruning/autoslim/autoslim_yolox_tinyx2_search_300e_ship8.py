_base_ = [
    './autoslim_yolox_tinyx2_300e_ship8.py',
]

algorithm = dict(distiller=None, input_shape=(3, 256, 320))

searcher = dict(
    type='GreedySearcher',
    target_flops=[int(x*1e9) for x in [0.5, 1, 1.5]],
    max_channel_bins=8,
    metrics='mAP',
    score_key='mAP')

data = dict(samples_per_gpu=64, workers_per_gpu=4)
