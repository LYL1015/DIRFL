import torchvision
import json
import shutil
from copy import deepcopy
from pickle import FALSE
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import os
import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import initialize_weights_new
from pytorch_lightning.loggers.wandb import WandbLogger
from compute_loss import Compute_loss
from models.models import MODELS
from utils.metrics_inference import *


# create floder
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
# select dataset
__dataset__ = {
               "wv3_dataset":Data,
               "GF2_dataset":Data,
               "wv2_dataset":Data
               }
        
class CoolSystem(pl.LightningModule):
    def __init__(self):
        """初始化训练的参数"""
        super(CoolSystem, self).__init__()
        # train datasets
        self.train_datasets = __dataset__[config["train_dataset"]](
                            config,
                            is_train=True,
                        )
        self.train_batchsize = config["train_batch_size"]
        # val datasets
        self.validation_datasets = __dataset__[config["train_dataset"]](
                            config,
                            is_train=False,
                        )
        self.val_batchsize = config["val_batch_size"]
        self.num_workers = config["num_workers"]

        # set mode stype
        self.model =  MODELS[config["model"]](config)
        # resume...
        if args.resume is not None:
            print("Loading from existing chekpoint and copying weights to continue....")
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint, strict=False)
        print(PATH)
        # print model summary.txt
        import sys
        original_stdout = sys.stdout 
        with open(PATH+"/"+"model_summary.txt", 'w+') as f:
            sys.stdout = f
            print(f'\n{self.model}\n')
            sys.stdout = original_stdout 
        shutil.copy(f'./models/{config["model"]}.py',PATH+"/"+"model.py") 
    def train_dataloader(self):
        train_loader = data.DataLoader(
                        self.train_datasets,
                        batch_size=self.train_batchsize,
                        num_workers=self.num_workers,
                        shuffle=True,
                        # pin_memory=False,
                    )
        return train_loader
    
    def val_dataloader(self):
        val_loader = data.DataLoader(
                        self.validation_datasets,
                        batch_size=self.val_batchsize,
                        num_workers=self.num_workers,
                        shuffle=False,
                        # pin_memory=False,
                    )
        return val_loader


    def configure_optimizers(self):
        """configuration optimizer and learning rate tuning strategies"""
        # setting up optimizer.
        self.initlr =config["optimizer"]["args"]["lr"] 
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.momentum = config["optimizer"]["args"]["momentum"]
        if config["optimizer"]["type"] == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.initlr , 
                momentum =self.momentum , 
                weight_decay= self.weight_decay
            )
        elif config["optimizer"]["type"] == "ADAM":
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.initlr,
                weight_decay= self.weight_decay,
                betas =  [0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "ADAMW":
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.initlr,
                weight_decay= self.weight_decay,
                betas =  [0.9, 0.999]
            )   
        else:
            exit("Undefined optimizer type")
        
        # learning rate tuning strategies
        if config["optimizer"]["sheduler"] == "StepLR":
            step_size=config["optimizer"]["sheduler_set"]["step_size"]
            gamma=config["optimizer"]["sheduler_set"]["gamma"]
            scheduler = optim.lr_scheduler.StepLR(  optimizer, step_size=step_size, gamma=gamma)
    
        return [optimizer], [scheduler]
    
            
    def forward(self,MS_image, PAN_image,image_dict):
        out = self.model(MS_image, PAN_image)
        return out
    
    def training_step(self, data):
        """trainning step"""
        # read data
        image_dict, MS_image, PAN_image, gt = data
        # taking model output
        out = self.forward(MS_image, PAN_image,image_dict)      
        # computing loss 
        loss = Compute_loss(config,out,PAN_image,gt)
        self.log('train_loss', loss,sync_dist=True,prog_bar=True)
        self.log('lr',self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr'],sync_dist=True,prog_bar=True)
        
        return {'loss': loss}
    
    def on_validation_epoch_start(self):
        self.pred_dic={}
        return super().on_validation_epoch_start()

    def validation_step(self, data, batch_idx):
        """validation step"""
        # read data.
        image_dict, MS_image, PAN_image, gt = data
        # taking model output
        out     = self.forward(MS_image, PAN_image,image_dict)   
        pred    = out['pred'];pred = torch.clip(pred,min=0,max=1)
        # computing loss 
        loss = Compute_loss(config,out,PAN_image,gt)
        # computing performance metrics
        max_value = config[config["train_dataset"]]["max_value"]
        if not config[config["train_dataset"]]['normalize']:
            predict_y = torch.clip((pred * max_value),min=0,max=max_value)
            ground_truth = torch.clip((gt * max_value),min=0,max=max_value)
        else:        
            predict_y = torch.clip(((pred+ 1) * max_value/2),min=0,max=max_value/2)
            ground_truth = torch.clip(((gt+ 1) * max_value/2),min=0,max=max_value/2)
                
        predict_y = np.array(predict_y.cpu().squeeze(0).permute(1,2,0))/max_value
        ground_truth = np.array(ground_truth.cpu().squeeze(0).permute(1,2,0))/max_value
        
        c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q = ref_evaluate(predict_y,ground_truth)

        self.log('val_loss', loss,sync_dist=True,prog_bar=True)
        self.log('psnr', c_psnr,sync_dist=True, prog_bar=True)
        self.log('ergas', c_ergas,sync_dist=True, prog_bar=True)
        self.log('sam',c_sam,sync_dist=True, prog_bar=True)

        self.trainer.checkpoint_callback.best_model_score #save the best score model
        return {'val_loss': loss, 'psnr': psnr,'ergas':ergas,'sam':sam}
    

def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/Train_DIRFL.json',type=str,
                            help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                            help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                            help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    global args
    args = parser.parse_args()
    
    global config
    config = json.load(open(args.config))
    
    # set seeds.
    seed = 123 #Global seed set to 42
    seed_everything(seed)
    
    # wandb log init
    global wandb_logger
    # import wandb
    wandb_logger = WandbLogger(project=config['name']+"-"+config["train_dataset"])
    
    # setting up path
    global PATH
    PATH = "./"+config["experim_name"]+"/"+config["train_dataset"]+"/"+str(config["tags"])
    ensure_dir(PATH+"/")
    shutil.copy2(args.config, PATH)

    # init pytorch-litening
    model = CoolSystem()
    
    # set checkpoint mode and init ModelCheckpointHook
    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    dirpath=PATH,
    filename='best_model-epoch:{epoch:02d}-psnr:{psnr:.4f}-ergas:{ergas:.4f}-sam:{sam:.4f}',
    auto_insert_metric_name=False,   
    every_n_epochs=config["trainer"]["test_freq"],
    save_on_train_epoch_end=True,
    save_top_k=6,
    mode = "max"
    )

    if args.resume is not  None:
        trainer = pl.Trainer(
            # strategy = "ddp",
            max_epochs=config["trainer"]["total_epochs"],
            resume_from_checkpoint = args.resume,
            accelerator='gpu', devices=[0],
            logger=wandb_logger,
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config["trainer"]["test_freq"],
        ) 
    else:
        trainer = pl.Trainer(
            # strategy = "ddp",
            max_epochs=config["trainer"]["total_epochs"],
            accelerator='gpu', devices=[0],
            logger=wandb_logger,
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config["trainer"]["test_freq"],
        )   
    trainer.fit(model)

if __name__ == '__main__':
    print('-----------------------------------------train_pl.py trainning-----------------------------------------')
    main()
    