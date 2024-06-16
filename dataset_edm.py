import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation
from utils.EDM import generate_edm


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        #存储图片文件
        self.img_file = self._select_img(self.data_file)
        
        # self.edm = self._make_edm(self.img_file)   #返回的是一个列表，每个元素是生成的edm torch的
        
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()
        
        dgt_file = "dgt" + img_file[3:]
        with open(file=os.path.join(self.data_path, dgt_file), mode='rb') as file:
            dgt = torch.from_numpy(pickle.load(file)).float()
        
        cgt_file = "cgt" + img_file[3:]
        with open(file=os.path.join(self.data_path, cgt_file), mode='rb') as file:
            cgt = torch.from_numpy(pickle.load(file)).float()
        
        # edm = self.edm[idx]
        
        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)
            torch.manual_seed(seed)
            dgt = self.transforms(dgt)
            torch.manual_seed(seed)
            cgt = self.transforms(cgt)
            # torch.manual_seed(seed)
            # edm = self.transforms(edm)

        # return img, gt, dgt, edm
        return img, gt, dgt, cgt
        

    #对于image来说 前三位是img   对于GT来说前2位是gt
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
        
        # print(img_list[0])
        # exit()
        # img_list.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))#按数字进行排序
        return img_list
    
    
    def _make_edm(self, img_file):
        edm = []
        for i in range(len(img_file)):
            gt_file = "gt" + img_file[i][3:]
            with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
                gt = torch.from_numpy(pickle.load(file)).float()
                edm.append(generate_edm(gt))
        
        return edm

    def __len__(self):
        return len(self.img_file)
