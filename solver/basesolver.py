
import os, torch, time
from Datasets.test_dataset import data
from Datasets.data import get_data
from torch.utils.data import DataLoader

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 1

        self.timestamp = int(time.time())

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0
        self.val_dataset = get_data(cfg, cfg['data_dir_eval'])
        self.val_loader = DataLoader(self.val_dataset, cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers,pin_memory=False,)
        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            self.epoch += 1
