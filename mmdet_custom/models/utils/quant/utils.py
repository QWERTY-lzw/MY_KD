import torch
import torch.nn as nn

def extract_names(m, target_class, conv2d_names=[], base_name=''):
    for name, child in m.named_children():
        now_name = base_name + '.' + name if base_name != '' else name
        if type(child) == target_class:
            conv2d_names.append(now_name)
        else:
            extract_names(child, target_class, conv2d_names, now_name)


def setattr_dot(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


def getattr_dot(model, name):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    return getattr(model, name_list[-1])
