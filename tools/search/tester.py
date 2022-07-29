# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np
import pickle
import shutil
import tempfile

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import single_gpu_test, multi_gpu_test

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

@no_grad_wrapper
def get_cand_map(model, args, distributed, cfg, test_data_loader, test_dataset):
    if not distributed: # False
        model = MMDataParallel(model, device_ids=[0])
        # bn_training_mode(model)
        outputs = single_gpu_test(model, test_data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        outputs = multi_gpu_test(model, test_data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info() # rank = 0
    if rank == 0:
        if args.out: # None
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            test_dataset.format_results(outputs, **kwargs)
        if args.eval: # bbox
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = test_dataset.evaluate(outputs, **eval_kwargs)

            # metric_dict = dict(config=args.config, metric=metric)
            print(metric)

            #OrderedDict([('bbox_mAP', 0.0), ('bbox_mAP_50', 0.0), ('bbox_mAP_75', 0.0), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.0), ('bbox_mAP_l', 0.0), ('bbox_mAP_copypaste', '0.000 0.000 0.000 0.000 0.000 0.000')])
            map = []
            for key in metric:
                map.append(metric[key])
            return tuple(map[:-1]) if len(map)>1 else tuple(map)# (0.6599323749542236,)

    return None
