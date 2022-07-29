# Copyright (c) OpenMMLab. All rights reserved.

from mmrazor.models.builder import ARCHITECTURES
from mmrazor.models.architectures.base import BaseArchitecture


@ARCHITECTURES.register_module()
class MyMMDetArchitecture(BaseArchitecture):
    """Architecture based on MMDet."""

    def __init__(self, **kwargs):
        super(MyMMDetArchitecture, self).__init__(**kwargs)

    def cal_pseudo_loss(self, pseudo_img):
        """Used for executing ``forward`` with pseudo_img."""
        out = 0.
        for levels in pseudo_img:
            out += sum([level.sum() for level in levels])

        return out
