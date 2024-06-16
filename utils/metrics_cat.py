import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score

batches=0
class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target, patch_size=None,threshold=None, predict_b=None):
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)
    row=584//patch_size+1
    colume=565//patch_size+1
    predict_b=torch.sigmoid(predict).cpu().detach().numpy()

    target = target.cpu().detach().numpy()
    predict_n=[]
    target_n=[]
    index=0
    for i in range(row*colume):
        if (i+1)%colume!=0 and i<colume*(row-1):
            predict_n.extend(predict_b[i].flatten().tolist())
            target_n.extend(target[i].flatten().tolist())
            index+=1
        if (i+1)%colume==0 and i!=row*colume-1:
            predict_n.extend(predict_b[i,:,:,0:565-(colume-1)*patch_size].flatten().tolist())
            target_n.extend(target[i,:,:,0:565-(colume-1)*patch_size].flatten().tolist())
            index+=1
        if i>=colume*(row-1) and i!=row*colume-1:
            predict_n.extend( predict_b[i,:,0:584-(row-1)*patch_size,:].flatten().tolist())
            target_n.extend( target[i,:,0:584-(row-1)*patch_size,:].flatten().tolist())
            index+=1
        if i==row*colume-1:
            predict_n.extend(predict_b[i,:,0:584-(row-1)*patch_size,0:565-(colume-1)*patch_size].flatten().tolist())
            target_n.extend(target[i,:,0:584-(row-1)*patch_size,0:565-(colume-1)*patch_size].flatten().tolist())           
            index+=1

    predict = np.array(predict_n)
    out=np.where(predict >= threshold, 1, 0)
    target = np.array(target_n)
    
    
    # #将predict_b的156个小图拼接成一张大图
    # predict_b_merged=predict_b.squeeze()
    # predict_b_row=[]
    # for i in range(row):
    #     predict_b_row.append(np.concatenate(predict_b_merged[i*colume:(i+1)*colume],axis=1))
    # predict_b_row=np.array(predict_b_row)
    # predict_b_all=np.concatenate(predict_b_row[0:row],axis=0)[0:584,0:565]    
    # print(predict_b_all[0].shape,"======================")
    # predict_b_all=np.where(predict_b_all >= threshold, 1, 0)
    global batches
    cv2.imwrite(
        f"/home/zjk/FR-UNet/saved/FR_UNet/save_patch/pre_patch{batches}.png", np.uint8(out*255))
    cv2.imwrite(
        f"/home/zjk/FR-UNet/saved/FR_UNet/save_patch/pre_patch{batches}.png",np.uint8(target*255))    
    batches+=1



    print("shape =========================",out.shape,target.shape)



    tp = (out * target).sum()
    tn = ((1 - out) * (1 - target)).sum()
    fp = ((1 - target) * out).sum()
    fn = ((1 - out) * target).sum()
    auc = roc_auc_score(target, predict)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }



def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n
