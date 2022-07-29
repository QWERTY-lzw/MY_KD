# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet_custom.visualization import imshow_det_bboxes as imshow_det_bboxes_and_points
from mmdet.datasets.builder import build_dataset
from mmdet.core.visualization import imshow_det_bboxes as imshow_det_bboxes

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg

def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
    img = mmcv.imread(img)
    img = img.copy()
    # img = img * 0
    if isinstance(result[0], tuple):
        bbox_result, keypoint_results = [r[0] for r in result], [r[1] for r in result]
        # if isinstance(segm_result, tuple):
        #     segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, keypoint_results = result, None
    if isinstance(bbox_result, list):
        bbox_result = bbox_result[0]

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    keypoints = None
    if keypoint_results is not None:
        if isinstance(keypoint_results, list):
            keypoint_results = keypoint_results[0]
        keypoints = np.vstack(keypoint_results)
        point_labels = [
        np.full(point.shape[0], i, dtype=np.int32)
            for i, point in enumerate(keypoint_results)
        ]
        point_labels = np.concatenate(point_labels)
        point_labels = point_labels + self.bbox_head.num_classes
        # keypoints = np.vstack(keypoint_results)
    # print(keypoints)
    # draw segmentation masks
    segms = None
    # if keypoints is not None and len(labels) > 0:  # non empty
    #     segms = mmcv.concat_list(segm_result)
    #     if isinstance(segms[0], torch.Tensor):
    #         segms = torch.stack(segms, dim=0).detach().cpu().numpy()
    #     else:
    #         segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes_and_points(
        img,
        bboxes,
        labels,
        segms,
        keypoints,
        point_labels,
        class_names=self.CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [
            p for p in cfg.data.train.pipeline if p['type'] != 'SegRescale'
        ]
    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    # Specify the path to model config and checkpoint file
    checkpoint_file = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0], 'latest.pth')

    from mmdet.apis import init_detector, inference_detector
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, checkpoint_file, device='cuda:0')

    for i, item in enumerate(dataset):
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        result = inference_detector(model, [item['img']])
        img = show_result(model, item['img'], result, bbox_color='citys', font_size=6, thickness=6)
        
        # gt_bboxes = item['gt_bboxes'].data.numpy()
        gt_bboxes = item['gt_bboxes']
        # gt_labels = item['gt_labels'].data.numpy()
        gt_labels = item['gt_labels']
        gt_masks = item.get('gt_masks', None)
        gt_keypoints = item.get('gt_keypoints', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)

        gt_seg = item.get('gt_semantic_seg', None)
        if gt_seg is not None:
            pad_value = 255  # the padding value of gt_seg
            sem_labels = np.unique(gt_seg)
            all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
            all_labels, counts = np.unique(all_labels, return_counts=True)
            stuff_labels = all_labels[np.logical_and(counts < 2,
                                                     all_labels != pad_value)]
            stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
            gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
            gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                      axis=0)
            # If you need to show the bounding boxes,
            # please comment the following line
            gt_bboxes = None
        # img = (item['img'].data.permute(1,2,0)*255).numpy().astype(np.uint8)
        # img = (item['img'].data.permute(1,2,0)).numpy().astype(np.uint8)
        # img = (item['img'].permute(1,2,0)).astype(np.uint8)
        # imshow_det_bboxes(
        #     img,
        #     gt_bboxes,
        #     gt_labels,
        #     gt_masks,
        #     # gt_keypoints,
        #     class_names=dataset.CLASSES,
        #     show=not args.not_show,
        #     wait_time=args.show_interval,
        #     out_file=filename,
        #     bbox_color=dataset.PALETTE,
        #     text_color=(200, 200, 200),
        #     mask_color=dataset.PALETTE)
        imshow_det_bboxes(
            img,
            gt_bboxes,
            gt_labels,
            gt_masks,
            # gt_keypoints,
            class_names=dataset.CLASSES,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=dataset.PALETTE,
            text_color=None,
            thickness=2,
            font_size=1,
            mask_color=dataset.PALETTE)

        progress_bar.update()


if __name__ == '__main__':
    main()
