import argparse
from bunch import Bunch
from loguru import logger
# from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset_edm import vessel_dataset
from trainer_tuning import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
import yaml

def main(CFG, data_path, batch_size, with_val=False):
    seed_torch()
    if with_val:
        train_dataset = vessel_dataset(data_path, mode="training", split=0.9)
        val_dataset = vessel_dataset(
            data_path, mode="training", split=0.9, is_val=True)
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")
        
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    logger.info('The patch number of train is %d' % len(train_dataset))
    
    #这里开始就要大改了  这是加载网络
    #这里的models是一个
    '''
    当您使用 import models 时，您导入的是 models 模块（或包），它指的是 models/__init__.py 文件中定义的内容。
    这个模块包含了各种模型类的定义，包括您之前提到的 FR_UNet 类。
    '''
    model1 = get_instance(models, 'model_dilation', CFG)
    model2 = get_instance(models, 'model_tuning', CFG)
    
    # logger.info(f'\n{model}\n')
    
    #这里也要改   好像也不用改，反正两个stage网络用的都是 BCE
    loss_bce = get_instance(losses, 'loss_BCE', CFG)
    loss_dice = get_instance(losses, 'loss_DICE', CFG)
    loss_wbce = get_instance(losses, 'loss_WBCE', CFG)
    
    
    #这里就要大改了，训练的类 和 具体的函数需要改   因为现在是两个网络
    trainer = Trainer(
        #具体而言可以先改这个model
        model1=model1,
        model2=model2,
        CFG=CFG,
        loss_bce=loss_bce,
        loss_dice=loss_dice,
        loss_wbce=loss_wbce,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=128,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()

    with open('config_two_stage.yaml', encoding='utf-8') as file:
        CFG = Bunch(yaml.safe_load(file))
        # CFG = yaml.safe_load(file)  # 为列表类型
    main(CFG, args.dataset_path, args.batch_size, args.val)
