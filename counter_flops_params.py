
import  sys
from thop import profile
import importlib, torch
import math
import time
from models.models import MODELS
import re, yaml, os  
import sys
import json

def get_config(cfg_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
       u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.') 
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg

test_config_path = './configs/'
model_name_list = ['DIRFL']
if __name__ == "__main__":
    for model_name in model_name_list :
        print(f'-------------------------------{model_name}----------------------------')
        net_name = model_name
        cfg = 'Train_'+model_name+'.json'
        config  = json.load(open(test_config_path+cfg))
        model =  MODELS[net_name](config)
        input = torch.randn(1, 4, 32, 32)
        input1 = torch.randn(1, 1, 128, 128)
        input2 = torch.randn(1, 4, 128, 128)

        print("The thop result")
        flops, params = profile(model, inputs=(input, input1))
        print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e5)))
        print('===========================================================================')