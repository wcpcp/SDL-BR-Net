import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer_tuning import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import os
import pickle
from PIL import Image

class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.len = len(test_loader)
        self.gt_load = []
        
        for i in range(20):
            img_file = 'gt_'+str(i)+'.pkl'
            with open(file=os.path.join("/root/FR-UNet-master/dataset/DRIVE/test_pro_normal", img_file), mode='rb') as file:
                self.gt_load.append(torch.from_numpy(pickle.load(file)).float())
        
        # for i in range(20):
        #     gt = Image.open(os.path.join("/root/FR-UNet-master/dataset/DRIVE/training/1st_manual", "{}_manual1.gif".format(i + 21)))
        #     # print(gt)
        #     gt = np.array(gt)
        #     self.gt_load.append(torch.from_numpy(gt//255).float())
        
            
        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        # print("the dataloader len ===============",self.len)
        # patch_num = self.len/20
        patch_num = 3250/20
        pres = [[] for _ in range(20)]
        count=0
        count_recovery=0
        tic = time.time()
        patch_counter = 0
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                # dgt = dgt.cuda(non_blocking=True)
                
                pre_d, pre_c, pre = self.model(img)                
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
                B,_,_,_ = pre.shape
                
                for i in range(B):
                    pres[count].append(pre[i].cpu().numpy())
                    patch_counter += 1
                    if patch_counter % (patch_num*4) == 0:
                        patch_counter=0
                        count += 1
                
                # pres.append(pre.cpu().numpy())
                # if patch_counter % patch_num == 0 and patch_num != 0:
                if count_recovery != count:
                
                    h=584
                    w=565
                    patch_h = self.CFG.patch_size
                    patch_w = self.CFG.patch_size
                    stride_h = self.CFG.patch_stride
                    stride_w = self.CFG.patch_stride
                    # print("=============",self.CFG.patch_size,"=",self.CFG.patch_stride)
                    img_h = stride_h - (h - patch_h) % stride_h + 584
                    img_w = stride_w - (w - patch_w) % stride_w + 565
                    # print((img_h - patch_h) // stride_h,"1============(",img_h,img_w,")")
                    # print((img_h - patch_h) // stride_h +1,"2============")
                    # N_patches_h = (img_h - patch_h) // stride_h + 1
                    # N_patches_w = (img_w - patch_w) // stride_w + 1
                    # N_patches_img = N_patches_h * N_patches_w
                    # 这个是记录对一个地方覆盖多个patch的总概率
                    full_prob = np.zeros((1, 1, img_h, img_w))
                    # 这个是记录对一个地方覆盖多个patch的总个数
                    full_sum = np.zeros((1, 1, img_h, img_w))

                    k = 0  # iterator over all the patches
                    for h in range((img_h - patch_h) // stride_h +1):
                        for w in range((img_w - patch_w) // stride_w +1):
                            full_prob[0, 0, h * stride_h:(h * stride_h) + patch_h,w * stride_w:(w * stride_w) + patch_w] += pres[count_recovery][k][0]  # Accumulate predicted values
                            # print("count_recovery=",count_recovery,"k=",k)
                            full_sum[0, 0, h * stride_h:(h * stride_h) + patch_h,w * stride_w:(w * stride_w) + patch_w] += 1  # Accumulate the number of predictions
                            k += 1
                    
                    
                    pre = full_prob / full_sum  # Take the average相除就得到了平均概率
                    pre=torch.from_numpy(pre)
                    # print(pre.shape,"============================================================")   #torch.Size([1, 1, 588, 570])
                    pre = TF.crop(pre, 0, 0, 584, 565)
                    gt = TF.crop(self.gt_load[count_recovery], 0, 0, 584, 565)
                    
                    # if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)

                    predict = Image.fromarray(np.uint8(predict[0][0]*255))
                    predict.save(f"/root/FR-UNet-master/save_picture/pre{count_recovery}.png")

                    predict_b = Image.fromarray(np.uint8(predict_b[0][0]*255))
                    predict_b.save(f"/root/FR-UNet-master/save_picture/pre_b{count_recovery}.png")

                    gt_img = Image.fromarray(np.uint8(gt.cpu().numpy()[0]*255))
                    gt_img.save(f"/root/FR-UNet-master/save_picture/gt{count_recovery}.png")

                    print("finish saving pictures")
                    
                    
                    if self.CFG.DTI:
                        pre_DTI = double_threshold_iteration(
                            i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                        self._metrics_update(
                            *get_metrics(pre, gt, predict_b=pre_DTI).values())
                        if self.CFG.CCC:
                            self.CCC.update(count_connect_component(pre_DTI, gt))
                    else:
                        self._metrics_update(
                            *get_metrics(pre, gt, self.CFG.threshold).values())
                        if self.CFG.CCC:
                            self.CCC.update(count_connect_component(
                                pre, gt, threshold=self.CFG.threshold))
                    
                    count_recovery = count_recovery+1
                    # pres.clear()

                    tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                    
                tic = time.time()
        
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
        