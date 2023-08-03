
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .test_dataset import Data, Data_test
from torchvision import transforms
import torch, numpy  #h5py, 
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    cfg = cfg
    return Data(data_dir_ms, data_dir_pan, cfg, transform=transform())
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform())
