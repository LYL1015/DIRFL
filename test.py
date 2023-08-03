
from solver.testsolver_o import Testsolver as Testsolver_o
from solver.testsolver import Testsolver as Testsolver1
import re, yaml, os  
import sys
from Test_Tool.demo_deep_methods import evaluate_metric


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

def save_yml(info, cfg_path):
    with open(cfg_path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.SafeDumper)

save_dir_GF = './result/GF2/'
save_dir_GF_rgb = './result/GF2_rgb/'

save_dir_WV2 = './result/WV2/'
save_dir_WV2_rgb = './result/WV2_rgb/'

save_dir_WV3 = './result/WV3/'
save_dir_WV3_rgb = './result/WV3_rgb/'

save_dir_fullGF2 = './result/fullGF2/'
save_dir_fullGF2_rgb = './result/fullGF2_rgb/'

test_config_path = './configs/'
list_algorithm = [
                {'algorithm':'DIRFL','model':'./Experiments/DIRFL/wv3_dataset/EXP1/DIRFL_best_model_epoch:1.pth',
                   'test_config_path':test_config_path+'Train_'+'DIRFL'+'.json'}
                  ]

dataset_selceted = ['WV3','WV2','GF2','fullGF2']
if __name__ == '__main__':
    cfg = get_config('./configs/Test_option.yml')
    for list in list_algorithm:
        # initialize
        cfg['algorithm']= list['algorithm'];cfg['test']['algorithm']= list['algorithm'];cfg['test2']['algorithm']= list['algorithm'];cfg['test3']['algorithm']= list['algorithm'];cfg['test4']['algorithm']= list['algorithm']
        cfg['test']['model']= list['model'];cfg['test']['test_config_path']= list['test_config_path']
        
        cfg['test']['save_dir'] = save_dir_GF+list['algorithm']+'/'
        cfg['test']['save_dir_rgb'] = save_dir_GF_rgb+list['algorithm']+'/'
        cfg['test']['save_dir_rgb'] = save_dir_GF_rgb+'/'


        cfg['test2']['model']= list['model'];cfg['test2']['test_config_path']= list['test_config_path']
        cfg['test2']['save_dir'] = save_dir_WV2+list['algorithm']+'/'
        cfg['test2']['save_dir_rgb'] = save_dir_WV2_rgb+list['algorithm']+'/'
        cfg['test2']['save_dir_rgb'] = save_dir_WV2_rgb+'/'
        
        cfg['test3']['model']= list['model'];cfg['test3']['test_config_path']= list['test_config_path']
        cfg['test3']['save_dir'] = save_dir_WV3+list['algorithm']+'/'
        cfg['test3']['save_dir_rgb'] = save_dir_WV3_rgb+list['algorithm']+'/'
        cfg['test3']['save_dir_rgb'] = save_dir_WV3_rgb+'/'

        cfg['test4']['model']= list['model'];cfg['test4']['test_config_path']= list['test_config_path']
        cfg['test4']['save_dir'] = save_dir_fullGF2+list['algorithm']+'/'
        cfg['test4']['save_dir_rgb'] = save_dir_fullGF2_rgb+list['algorithm']+'/'
        cfg['test4']['save_dir_rgb'] = save_dir_fullGF2_rgb+'/'

        
        # GF2
        if 'GF2' in dataset_selceted:
          method = cfg['algorithm'];data_type = cfg['test']['datatype']
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++')
          solver = Testsolver1(cfg)
          solver.run()
          evaluate_metric(cfg)
          cfg['test']['save_dir'] = cfg['test']['save_dir_rgb']
          solver = Testsolver_o(cfg)
          solver.run()
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++\n')

        # WV2
        if 'WV2' in dataset_selceted:
          method = cfg['algorithm'];data_type = cfg['test2']['datatype']
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++')
          temp = cfg['test']
          cfg['test'] = cfg['test2']
          solver = Testsolver1(cfg)
          solver.run()
          evaluate_metric(cfg)
          cfg['test2']['save_dir'] = cfg['test2']['save_dir_rgb']
          solver = Testsolver_o(cfg)
          solver.run()
          cfg['test'] = temp
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++\n')

        
       # WV3
        if 'WV3' in dataset_selceted:
          method = cfg['algorithm'];data_type = cfg['test3']['datatype']
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++')
          temp = cfg['test']
          cfg['test'] = cfg['test3']
          solver = Testsolver1(cfg)
          solver.run()
          evaluate_metric(cfg)
          cfg['test3']['save_dir'] = cfg['test3']['save_dir_rgb']
          solver = Testsolver_o(cfg)
          solver.run()
          print("generate rgb result")
          cfg['test'] = temp
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++\n')

        if 'fullGF2' in dataset_selceted:
          method = cfg['algorithm'];data_type = cfg['test4']['datatype']
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++')
          temp = cfg['test']
          cfg['test'] = cfg['test4']
          solver = Testsolver1(cfg)
          solver.run()
          evaluate_metric(cfg)
          cfg['test4']['save_dir'] = cfg['test4']['save_dir_rgb']
          solver = Testsolver_o(cfg)
          solver.run()
          cfg['test'] = temp
          print(f'+++++++++++++++++++++++++++++++++{method} testting : {data_type}+++++++++++++++++++++++++++++++++\n')

        print(f'==================================================================================================')
        print(f'==================================================================================================')



    
    
