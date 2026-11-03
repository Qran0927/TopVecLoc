# Copyright (c) Meta Platforms, Inc. and affiliates.            # （/）

from copy import deepcopy                                        #  deepcopy：（）
from pathlib import Path                                         # Path：
from typing import Any, Dict, List                               # ：Any/Dict/List（ Dict[str, Any]）
# from logger import logger                                      # （）
import numpy as np                                               #  numpy（）
# import torch                                                   #  torch（ DataLoader  torch.utils.data）
# import torch.utils.data as torchdata                           # ：torchdata （）
# import torchvision.transforms as tvf                            # ：transform （ tvf）
from omegaconf import DictConfig, OmegaConf                      # OmegaConf：（hydra/omegaconf ）
import pytorch_lightning as pl                                   # PyTorch Lightning：/
from dataset.UAV.dataset import UavMapPair                        # ：UavMapPair（+）
# from torch.utils.data import Dataset, DataLoader               # ：Dataset/DataLoader（）
# from torchvision import transforms                             # ：transforms（ torchvision.transforms as tvf）
from torch.utils.data import Dataset, ConcatDataset               # Dataset：；ConcatDataset：
from torch.utils.data import Dataset, DataLoader, random_split    # DataLoader：；random_split：（）
from dataset.torch import collate                                  #  collate： graph 
import torchvision.transforms as tvf                              # tvf：torchvision /（ToTensor/Resize/ColorJitter ）

# ，pl.LightningDataModule                 # Lightning  DataModule  train/val/test 
class UavMapDatasetModule(pl.LightningDataModule):


    def __init__(self, cfg: Dict[str, Any]):                     # ： cfg （ OmegaConf ）
        super().__init__()                                       #  LightningDataModule 


        self.cfg=cfg                                              # （cfg）

        tfs = []                                                 # ， transforms（/）
        tfs.append(tvf.ToTensor())                                # 1)  Tensor（ [0,1] ）
        tfs.append(tvf.Resize(self.cfg.image_size))               # 2) Resize  cfg.image_size（ 256x256）
        self.val_tfs = tvf.Compose(tfs)                           #  transform： ToTensor + Resize（）

        if cfg.augmentation.image.apply:                          # （）
            args = OmegaConf.masked_copy(                         #  cfg.augmentation.image （ apply ）
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))                   # 3) ：（///）
        self.train_tfs = tvf.Compose(tfs)                         #  transform：ToTensor + Resize + ()ColorJitter

        self.init()                                               #  train_dataset / val_dataset（）

    def init(self):                                               # （ UavMapPair ）
        self.train_dataset = ConcatDataset([                      # 
            UavMapPair(root=Path(self.cfg.root),                  #  UavMapPair：
                      city=city,                                  #   /
                      training=True,                               #   training=True：（ train list  train ）
                      transform=self.train_tfs)                    #    transform（）
            for city in self.cfg.train_citys                      #  cfg.train_citys（）
        ])

        self.val_dataset = ConcatDataset([                        # 
            UavMapPair(root=Path(self.cfg.root),                  #  UavMapPair：
                      city=city,                                  #   
                      training=False,                              #   training=False：/
                      transform=self.val_tfs)                      #    transform（）
            for city in self.cfg.val_citys                        #  cfg.val_citys（）
        ])

        # self.val_datasets = {                                   # ：： val dataset（）
        #     city:UavMapPair(root=Path(self.cfg.root),city=city,transform=self.val_tfs)
        #     for city in self.cfg.val_citys
        #     }
        # logger.info("train data len:{},val data len:{}".format(len(self.train_dataset),len(self.val_dataset)))
        #                                                       # ：/
        # #                                       # ： dataset， train/val
        # train_ratio = 0.8  #                        # ： 0.8
        # #                                   # ： train/val 
        # train_size = int(len(self.dataset) * train_ratio)       # ：
        # val_size = len(self.dataset) - train_size               # ：
        # self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        #                                                       # ：random_split （）

    def train_dataloader(self):                                   # Lightning  DataLoader
        train_loader = DataLoader(self.train_dataset,             #  train_dataset  DataLoader
                                  batch_size=self.cfg.train.batch_size,   # batch （cfg.train.batch_size）
                                  num_workers=self.cfg.train.num_workers, # （Windows /）
                                  shuffle=True,                   # 
                                  pin_memory=True,
                                  collate_fn=collate)             #  collate： graph 
        return train_loader                                       #  DataLoader

    def val_dataloader(self):                                     # Lightning  DataLoader
        val_loader = DataLoader(self.val_dataset,                 #  val_dataset  DataLoader
                                batch_size=self.cfg.val.batch_size,       #  batch 
                                num_workers=self.cfg.val.num_workers,     #  num_workers
                                shuffle=False,                    # 
                                pin_memory=True,
                                collate_fn=collate)               #  collate： graph 

        return val_loader                                         #  DataLoader
