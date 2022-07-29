import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class FGDPROLoss(nn.Module):

    """PyTorch version of `Feature Distillation for General Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.0005
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 alpha_fgd=0.001):
        super(FGDPROLoss, self).__init__()
        #self.temp = temp
        self.alpha_fgd = alpha_fgd
        
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1))

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                scale_y,
                scale_x,
                loss_name,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_fgd
            
        return loss


    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        new_fea = self.generation(preds_S)
        #import ipdb;ipdb.set_trace()
        dis_loss = loss_mse(new_fea, preds_T)/N
        #dis_loss = loss_mse(preds_S,preds_T)/N
        return dis_loss
