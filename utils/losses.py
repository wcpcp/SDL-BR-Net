import torch
import torch.nn as nn
import torch.nn.functional as F
from .EDM import generate_edm
import numpy as np

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#         self.CE_loss = nn.CrossEntropyLoss(
#             reduction=reduction, ignore_index=ignore_index, weight=alpha)

#     def forward(self, output, target):
#         logpt = self.CE_loss(output, target)
#         pt = torch.exp(-logpt)
#         loss = ((1 - pt) ** self.gamma) * logpt
#         if self.reduction == 'mean':
#             return loss.mean()
#         return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2, logits = True, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Focal_Loss(nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self,alpha=0.25,gamma=2):
        super(Focal_Loss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self,preds,labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        preds = self.sigmoid(preds)
        eps=1e-7
        loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
        loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
        loss=loss_0+loss_1
        return torch.mean(loss)

    
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, targets):
        return self.mse_loss(prediction, targets)


class MSELossWithLogits(nn.Module):
    def __init__(self, reduction='sum'):
        super(MSELossWithLogits, self).__init__()
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def forward(self, prediction, targets, mask):
        prediction = self.sigmoid(prediction)     #0-1的概率图 
        # n = torch.count_nonzero(mask)
        # loss = (prediction*mask - targets*mask) ** 2  #l2损失
        # loss = torch.abs(prediction - targets)   #l1损失
        loss = F.l1_loss(prediction, targets, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

            
# class MSELossWithLogits(nn.Module):
#     def __init__(self, reduction='mean'):
#         super(MSELossWithLogits, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.mse_loss = nn.MSELoss(reduction=reduction)
        

#     def forward(self, prediction, targets, mask):
#         prediction = self.sigmoid(prediction)     #0-1的概率图  
#         n = torch.count_nonzero(mask)
#         #但是target是灰度化的原图   要实现复原操作   但是这样算损失函数肯定不对  因为target是0-255的
#         return self.mse_loss(prediction*mask, targets*mask)
    
class edm_loss(nn.Module):
    def __init__(self, reduction=None, pos_weight=1.0):
        super(edm_loss, self).__init__()
        self.generate_edm = generate_edm
        self.sigmoid = nn.Sigmoid()
        self.focal = FocalLoss()
        
    def forward(self, pred, y, edm):
        p = pred
        pred = self.sigmoid(pred)
        # edm = self.generate_edm(y)
        
        # 前景损失
        loss_fg = 1 - torch.sum(pred[y == 1] * edm[y == 1]) / torch.sum(y == 1)
        # 背景损失
        loss_bg = 1 * torch.sum(pred[y == 0] * edm[y == 0]) / torch.sum(y == 0)
        # 边缘损失
        edge_loss = loss_fg + loss_bg
        
        return edge_loss + 2*self.focal(p,y)


class BCELossWithWeights(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCELossWithWeights, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.generate_edm = generate_edm
        self.reduction = reduction

    def forward(self, pred, y, edm):
        # edm = self.generate_edm(y)
        # 计算原始的二分类交叉熵损失
        loss = self.bce_loss(pred, y)
        # 使用权重矩阵调整损失
        weighted_loss = loss * edm
        
        
        # 计算平均损失或者求和损失
        if self.reduction == "mean":
            return torch.mean(weighted_loss)
        elif self.reduction == "sum":
            return torch.sum(weighted_loss)
        else:
            return weighted_loss
        

class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        
        return self.bce_loss(prediction, targets)
        


class CELoss(nn.Module):
    def __init__(self, weight=[1, 1], ignore_index=-100, reduction='mean'):
        super(CELoss, self).__init__()
        weight = torch.tensor(weight).cuda()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target.squeeze(1).long())
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)
