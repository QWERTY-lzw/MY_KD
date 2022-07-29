#coding=utf-8
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn

import onnx
from onnx import helper
from onnx import numpy_helper

from mmcv.runner import load_checkpoint
from mmcv import Config
from mmdet.core.export import preprocess_example_input

from utils import approx_scale_as_mult_and_shift


parser = argparse.ArgumentParser(description="onnx2dat")
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument("onnx_file", type=str)
parser.add_argument('--output-file', type=str, default='tmp.dat')

def search(l, k, v):
    res = []
    for idx, info in enumerate(l):
        if v in info[k]:
            res.append(idx)
    return res

def del_items(l, k, v):
    res = []
    i = 0
    for idx, info in enumerate(l):
        if info[k] != v:
            info['idx'] = i
            i += 1
            res.append(info)
    return res

def print_items(l):
    print('*******************')
    for idx, info in enumerate(l):
        for k, v in info.items():
            if k in ['idx', 'type', 'input_idx_rel']:
                print('\'{}\': {}, '.format(k, v), end='')
        print('')

def check_layers(l):
    bad_list = []
    for info in l:
        idx = info['idx']
        ok = True
        if info['type'] == 'Conv':
            if 'extra_params' not in info:
                ok = False
            else:
                if 'min_yq' not in info['extra_params'] or \
                    'max_yq' not in info['extra_params'] or \
                    'weight_int' not in info['extra_params'] or \
                    'bias_int' not in info['extra_params'] or \
                    'A' not in info['extra_params'] or \
                    'N' not in info['extra_params']:
                    ok = False
        elif info['type'] in ['Concat', 'Add']:
            if 'extra_params' not in info:
                ok = False
            else:
                if 'min_yq' not in info['extra_params'] or \
                    'max_yq' not in info['extra_params']:
                    ok = False
        if not ok:
            bad_list.append(info)

        for i in info['input_idx']:
            if idx not in l[i]['output_idx']:
                ok = False
        for i in info['output_idx']:
            if idx not in l[i]['input_idx']:
                ok = False
    return bad_list

def main():
    args = parser.parse_args()
    if not args.onnx_file.endswith('.onnx'):
        print('Please Check Your ONNX Model Path Format')
        exit(0)
    onnx_model = onnx.load(args.onnx_file)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    all_params = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    layers = []
    for i, node in enumerate(onnx_model.graph.node):
        params = [all_params[par_name] for par_name in node.input if par_name in all_params]
        inputs = []
        for inp in node.input:
            if inp not in all_params.keys():
                inputs.append(inp)
        info = {'idx': i, 'name': node.name, 'id': int(node.name.split('_')[-1]),'type': node.op_type,
                'attribute': node.attribute, 'input': inputs, 'output': node.output, 'on': True,
                'params': params, 'extra_params': {}}
        layers.append(info)

    # fuse quant weight
    for info in layers:
        if info['type'] == 'Round' and len(info['input']) == 0:
            info['on'] = False
            assert len(info['params']) == 1
            weight_int = numpy_helper.to_array(info['params'][0]).round().astype('int8')

            assert len(info['output']) == 1
            next_clip_idx = search(layers, 'input', info['output'][0])
            assert len(next_clip_idx) == 1
            next_clip_idx = next_clip_idx[0]
            layers[next_clip_idx]['on'] = False
            assert len(layers[next_clip_idx]['params']) == 2
            min_wq = numpy_helper.to_array(layers[next_clip_idx]['params'][0]).astype('int32')
            max_wq = numpy_helper.to_array(layers[next_clip_idx]['params'][1]).astype('int32')
            weight_int = np.clip(weight_int, min_wq, max_wq)

            assert len(layers[next_clip_idx]['output']) == 1
            next_mul_idx = search(layers, 'input', layers[next_clip_idx]['output'][0])
            assert len(next_mul_idx) == 1
            next_mul_idx = next_mul_idx[0]
            layers[next_mul_idx]['on'] = False
            assert len(layers[next_mul_idx]['params']) == 1
            scale_w = numpy_helper.to_array(layers[next_mul_idx]['params'][0]).reshape(-1)

            cur_idx = next_mul_idx
            while layers[cur_idx]['type'] != 'Conv':
                layers[cur_idx]['on'] = False
                assert len(layers[cur_idx]['output']) == 1
                next_idx = search(layers, 'input', layers[cur_idx]['output'][0])
                assert len(next_idx) == 1
                next_idx = next_idx[0]
                layers[next_idx]['input'].remove(layers[cur_idx]['output'][0])
                cur_idx = next_idx

            conv_idx = cur_idx
            assert len(layers[conv_idx]['params']) == 1
            bias = numpy_helper.to_array(layers[conv_idx]['params'][0]).reshape(-1)
            extra_params = {'weight_int': weight_int, 'bias': bias, 'scale_w': scale_w,
                            'min_wq': min_wq, 'max_wq': max_wq}
            layers[conv_idx]['extra_params'].update(extra_params)

    # del Constant (all constant values are used on Slice)
    for info in layers:
        if info['type'] == 'Constant':
            info['on'] = False
    layers = del_items(layers, 'on', False)

    # fuse quant act
    for info in layers:
        if info['type'] == 'Relu':
            assert len(info['output']) == 1
            next_div_idx = search(layers, 'input', info['output'][0])
            assert len(next_div_idx) == 1
            next_div_idx = next_div_idx[0]
            layers[next_div_idx]['on'] = False
            assert len(layers[next_div_idx]['params']) == 1
            scale_y = numpy_helper.to_array(layers[next_div_idx]['params'][0])

            assert len(layers[next_div_idx]['output']) == 1
            next_round_idx = search(layers, 'input', layers[next_div_idx]['output'][0])
            assert len(next_round_idx) == 1
            next_round_idx = next_round_idx[0]
            layers[next_round_idx]['on'] = False

            assert len(layers[next_round_idx]['output']) == 1
            next_clip_idx = search(layers, 'input', layers[next_round_idx]['output'][0])
            assert len(next_clip_idx) == 1
            next_clip_idx = next_clip_idx[0]
            layers[next_clip_idx]['on'] = False
            assert len(layers[next_clip_idx]['params']) == 2
            min_yq = numpy_helper.to_array(layers[next_clip_idx]['params'][0]).astype('int32')
            max_yq = numpy_helper.to_array(layers[next_clip_idx]['params'][1]).astype('int32')

            assert len(layers[next_clip_idx]['output']) == 1
            next_mul_idx = search(layers, 'input', layers[next_clip_idx]['output'][0])
            assert len(next_mul_idx) == 1
            next_mul_idx = next_mul_idx[0]
            layers[next_mul_idx]['on'] = False

            info['output'] = layers[next_mul_idx]['output']
            extra_params = {'scale_y': scale_y, 'min_yq': min_yq, 'max_yq': max_yq}
            info['extra_params'].update(extra_params)

    layers = del_items(layers, 'on', False)

    # fuse quant act for concat and add
    for info in layers:
        if info['type'] in ['Concat', 'Add']:
            if len(info['output']) == 1 and len(info['input']) == 2:
                oup_relu_idx = search(layers, 'input', info['output'][0])
                assert len(oup_relu_idx) == 1
                oup_relu_idx = oup_relu_idx[0]
                info['extra_params'].update(layers[oup_relu_idx]['extra_params'])

                is_found = False
                key_idx = -1
                inp1_idx = search(layers, 'output', info['input'][0])
                while not is_found and len(inp1_idx) > 0:
                    if len(inp1_idx) > 1:
                        print("Warning! During getting the pre scale_y on \'{}\', detect multiple back path!".format(info['name']))
                    next_inp1_idx = []
                    for idx in inp1_idx:
                        if 'scale_y' in layers[idx]['extra_params']:
                            is_found = True
                            key_idx = idx
                        else:
                            for inp in layers[idx]['input']:
                                next_inp1_idx += search(layers, 'output', inp)
                    inp1_idx = next_inp1_idx

                if is_found:
                    layers[key_idx]['extra_params']['scale_y'] = layers[oup_relu_idx]['extra_params']['scale_y']
                else:
                    print("Error! No scale_x1 for \'{}\'!".format(info['name']))

                is_found = False
                key_idx = -1
                inp2_idx = search(layers, 'output', info['input'][1])
                while not is_found and len(inp2_idx) > 0:
                    if len(inp2_idx) > 1:
                        print("Warning! During getting the pre scale_y on \'{}\', detect multiple back path!".format(info['name']))
                    next_inp2_idx = []
                    for idx in inp2_idx:
                        if 'scale_y' in layers[idx]['extra_params']:
                            is_found = True
                            key_idx = idx
                        else:
                            for inp in layers[idx]['input']:
                                next_inp2_idx += search(layers, 'output', inp)
                    inp2_idx = next_inp2_idx

                if is_found:
                    layers[key_idx]['extra_params']['scale_y'] = layers[oup_relu_idx]['extra_params']['scale_y']
                else:
                    print("Error! No scale_x2 for \'{}\'!".format(info['name']))

            else:
                print("Warning! Wrong layer \'{}\', detect multiple inputs (>2)!".format(info['name']))

    # fuse relu to conv
    for info in layers:
        if info['type'] == 'Relu':
            assert len(info['output']) == 1
            inp = info['input']
            assert len(inp) == 1
            conv_idx = search(layers, 'output', inp[0])
            assert len(conv_idx) == 1
            conv_idx = conv_idx[0]
            info['on'] = False
            layers[conv_idx]['output'] = info['output']
            layers[conv_idx]['extra_params'].update(info['extra_params'])
    layers = del_items(layers, 'on', False)

    # get scale_x (find pre conv and scale_x = pre_scale_y)
    for info in layers:
        if info['type'] == 'Conv':
            pre_idx = []
            for inp in info['input']:
                pre_idx += search(layers, 'output', inp)
            is_found = False
            key_idx = -1
            while not is_found and len(pre_idx) > 0:
                if len(pre_idx) > 1:
                    print("Warning! During getting the pre scale_y on \'{}\', detect multiple back path!".format(info['name']))
                next_pre_idx = []
                for idx in pre_idx:
                    if 'scale_y' in layers[idx]['extra_params']:
                        is_found = True
                        key_idx = idx
                    else:
                        for inp in layers[idx]['input']:
                            next_pre_idx += search(layers, 'output', inp)
                pre_idx = next_pre_idx

            if is_found:
                scale_x = layers[key_idx]['extra_params']['scale_y']
                info['extra_params']['scale_x'] = scale_x
            else:
                info['extra_params']['scale_x'] = 1 / 255
                print("Warning! Set default scale for \'{}\'!".format(info['name']))

    # delete Focus and set input to first concat (only test on mmdet)
    for info in layers:
        if info['type'] == 'Slice':
            info['on'] = False
    layers = del_items(layers, 'on', False)
    for info in layers:
        if info['type'] == 'Concat' and len(info['input']) == 4:
            info['input'] = ['input']
            info['type'] = 'Focus'
            break
    
    for info in layers:
        if info['type'] == 'MaxPool':
            info['on'] = False
    layers = del_items(layers, 'on', False)
    for info in layers:
        if info['type'] == 'Concat' and len(info['input']) == 4:
            info['input'] = info['input'][:1]
            info['type'] = 'SPP'

    # get default scale_y
    for info in layers:
        if info['type'] == 'Conv':
            if 'scale_y' not in info['extra_params']:
                info['extra_params']['scale_y'] = 1
            if 'min_yq' not in info['extra_params']:
                info['extra_params']['min_yq'] = -2147483648
            if 'max_yq' not in info['extra_params']:
                info['extra_params']['max_yq'] = 2147483647

    # get A, N
    for info in layers:
        if info['type'] == 'Conv':
            extra_params = info['extra_params']
            scale_total = (extra_params['scale_x'] * extra_params['scale_w'] / extra_params['scale_y'])
            bias_int = (extra_params['bias'] / (extra_params['scale_w'] * extra_params['scale_x'])).round().astype('int32')
            A, N = approx_scale_as_mult_and_shift(scale_total, 8)
            updated_params = {'scale_total': scale_total, 'A': A, 'N': N, 'bias_int': bias_int}
            info['extra_params'].update(updated_params)

    # build input output idx
    for info in layers:
        input_idx = [search(layers, 'output', inp) for inp in info['input']]
        input_idx = sum(input_idx, [])
        info['input_idx'] = input_idx
        info['input_idx_rel'] = [i - info['idx'] for i in input_idx]

        output_idx = [search(layers, 'input', oup) for oup in info['output']]
        output_idx = sum(output_idx, [])
        info['output_idx'] = output_idx

    # check
    bad_list = check_layers(layers)
    if len(bad_list) > 0:
        print('Bad layers:')
        print_items(bad_list)
    else:
        print('All layers ok!')

    print_items(layers)

    # prepare input data
    input_config = {
        'input_shape': (1, 3, 256, 320),
        'input_path': osp.join(osp.dirname(__file__), '../../demo/demo.jpg')
    }
    one_img, one_meta = preprocess_example_input(input_config)

    # dat
    from test_dat import build_torch_layers
    torch_layers = build_torch_layers(layers)

    out_buffer = {}
    out_list1 = []
    for info in layers:
        idx = info['idx']
        input_idx = info['input_idx']
        output_idx = info['output_idx']
        if len(input_idx) == 0:
            inp = one_img
        else:
            inp = [out_buffer[i] for i in input_idx]
            if len(inp) == 1:
                inp = inp[0]

        oup = torch_layers[idx](inp)
        out_buffer[idx] = oup
        if len(output_idx) == 0:
            out_list1.append(oup)

    # mmdet
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build model
    cfg.model.train_cfg = None
    from mmdet_custom.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmdet.datasets import DATASETS
        dataset = DATASETS.get(cfg.data.test['type'])
        assert (dataset is not None)
        model.CLASSES = dataset.CLASSES
    model.cpu().eval()

    # Fuse bn
    from utils import fuse_conv_bn
    fuse_conv_bn(model)

    model.forward = model.forward_dummy
    out = model(one_img)
    out_list2 = []
    for i in range(3):
        for j in range(3):
            out_list2.append(out[j][i])

    from utils import lp_loss
    for i in range(9):
        dat_out = out_list1[i]
        torch_out = out_list2[i]
        loss = lp_loss(dat_out, torch_out)
        print(loss)

    import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    main()
