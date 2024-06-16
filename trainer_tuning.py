import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration, get_optimizer_instance
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
from PIL import Image

np.set_printoptions(threshold=np.inf)

class Trainer:
    def __init__(self, model1, model2, CFG=None, loss_bce=None, loss_dice=None, loss_wbce=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss_bce = loss_bce
        self.loss_dice = loss_dice
        self.loss_wbce = loss_wbce
        
        self.model1 = nn.DataParallel(model1.cuda())
        self.model2 = nn.DataParallel(model2.cuda())
        
        # 加载预训练模型权重
        if self.CFG.pretrained_model1_path:
            self.model1.load_state_dict(torch.load(self.CFG.pretrained_model1_path)['state_dict'])
        # if self.CFG.pretrained_model2_path:
        #     self.model2.load_state_dict(torch.load(self.CFG.pretrained_model2_path)['state_dict'])
        
        # 冻结模型1的参数
        if self.CFG.freeze_model1:
            for param in self.model1.parameters():
                param.requires_grad = False
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
#         self.optimizer = get_optimizer_instance(
#             torch.optim, "optimizer", CFG, 
#             list(self.model1.parameters()) + list(self.model2.parameters()) )  
        
        params = list(self.model2.parameters())
        if not self.CFG.freeze_model1:
            params += list(self.model1.parameters())

        self.optimizer = get_optimizer_instance(
            torch.optim, "optimizer", CFG, params)  # Only optimize parameters of model2
        
        # self.lr_scheduler = get_instance(
        #     torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        
        self.checkpoint_dir = os.path.join(CFG.save_dir, "two_stage", start_time)

        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        
        dir_exists(self.checkpoint_dir)
        
        cudnn.benchmark = True
        
        self.min_train_auc = -1

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            self._reset_metrics()
            self._train_epoch(epoch)
            # break
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            #可以通过这个save_period参数   来调整多少次epoch保存一个pth
            if epoch >= 30 and epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)
            
            if self.min_train_auc < self.auc.average:
                self._save_checkpoint_best(epoch)
                self.min_train_auc = self.auc.average

    def _train_epoch(self, epoch):
        self.model1.train()
        self.model2.train()
        
        #在这里，wrt_mode = 'train' 意味着你正在为训练过程创建SummaryWriter对象。
        wrt_mode = 'train'
        # self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()
        # for img, gt, mask in tbar:
        for img, gt, dgt, edm in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            dgt = dgt.cuda(non_blocking=True)
            edm = edm.cuda(non_blocking=True)
            
            self.optimizer.zero_grad()
        
            #amp是加速训练用的
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre1= self.model1(img)
                    pre = self.model2(pre1,img)
                    
                    # loss1 = self.loss_bce(pre, gt)
                    # loss2 = self.loss_dice(pre, gt)
                    # loss3 = self.loss_wbce(pre, gt, edm)
                    # loss = 2 * loss2 + loss3
                    loss = self.loss_bce(pre,gt)
                    # loss = 0.2*loss1 + 0.8*self.loss(pre, gt)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre1 = self.model1(img)
                pre = self.model2(torch.cat((pre1, img), dim=1))
                loss1 = self.loss(pre1, dgt)
                # loss = 0.2*loss1 + 0.8*self.loss(pre, gt)
                loss = loss1 + self.loss(pre, gt)
                
                loss.backward()
                self.optimizer.step()
                
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
            
        self.writer.add_scalar(f'{wrt_mode}/loss', self.total_loss.average, epoch)
        
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        #遍历optimizer1的参数组
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        # self.lr_scheduler.step()
        
        

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                else:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model2).__name__,
            'epoch': epoch,
            'state_dict1': self.model1.state_dict(),
            'state_dict2': self.model2.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename
    
    def _save_checkpoint_best(self, epoch):
        state = {
            'arch': type(self.model2).__name__,
            'epoch': epoch,
            'state_dict1': self.model1.state_dict(),
            'state_dict2': self.model2.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_best.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    #这些属性通常用于存储和计算训练或验证过程中的各种度量指标。
    #AverageMeter 是一个常用的工具类，用于计算和跟踪一个值的平均值和总和。
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
    # def _metrics_update(self, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
