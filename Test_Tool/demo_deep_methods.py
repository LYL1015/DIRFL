

import numpy as np
import cv2
import os
from scipy import signal
import re, yaml, os  
import sys

from Test_Tool.metrics import ref_evaluate, no_ref_evaluate
from PIL import Image
import argparse




        
def cal(ref, noref):
    reflist = []
    noreflist = []

    reflist.append(np.mean([ii[0] for ii in ref]))
    reflist.append(np.mean([ii[1] for ii in ref]))
    reflist.append(np.mean([ii[2] for ii in ref]))
    reflist.append(np.mean([ii[3] for ii in ref]))
    reflist.append(np.mean([ii[4] for ii in ref]))
    reflist.append(np.mean([ii[5] for ii in ref]))

    noreflist.append(np.mean([ih[0] for ih in noref]))
    noreflist.append(np.mean([ih[1] for ih in noref]))
    noreflist.append(np.mean([ih[2] for ih in noref]))
    return reflist, noreflist


def evaluate_metric(cfg1):
    list_ref = []
    list_noref = []
    cfg = cfg1

    datatype =cfg['test']['datatype']#GF2 WV2 WV3
    method  = cfg['test']['algorithm']
    test_data = cfg['test']['test_data']
    path_ms = f"./Data/{datatype}_data/{test_data}/ms"
    path_pan = f".Data/{datatype}_data/{test_data}/pan"
    path_predict =f"./result/{datatype}/{method}/test"
    # print(datatype)
    list_name = []
    for file_path in os.listdir(path_ms):
        list_name.append(file_path)
    #print("name---------------")
    #print(list_name)
    num = len(list_name)
    #num=2
    fnb = 0
    list_max = ''
    max_psnr =0
    scale = cfg['data']['upsacle']
    for file_name_i in list_name:
        '''loading data'''
        fnb = fnb+1
        path_ms_file = os.path.join(path_ms, file_name_i)
        path_pan_file = os.path.join(path_pan, file_name_i)
        path_predict_file = os.path.join(path_predict, file_name_i)

        original_msi = np.array(Image.open(path_ms_file))
        original_pan = np.array(Image.open(path_pan_file))
        fused_image = np.array(Image.open(path_predict_file))
        gt = np.uint8(original_msi)

        if datatype == 'fullGF2':
            used_ms = original_msi
        else:
            used_ms = cv2.resize(original_msi, (original_msi.shape[1]//scale, original_msi.shape[0]//scale), cv2.INTER_CUBIC)
        used_pan = np.expand_dims(original_pan, -1)

        '''setting save parameters'''



        '''evaluating all methods'''
        ref_results={}
        ref_results.update({'metrics: ':'  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q'})
        no_ref_results={}
        no_ref_results.update({'metrics: ':'  D_lamda,  D_s,    QNR'})

        '''Bicubic method'''
        #print('evaluating Bicubic method')

        #print(gt.shape)
        if datatype != 'fullGF2':
            temp_ref_results1 = ref_evaluate(fused_image, gt)
            list_ref.append(temp_ref_results1)
        temp_no_ref_results1 = no_ref_evaluate(fused_image, np.uint8(used_pan), np.uint8(used_ms))
        list_noref.append(temp_no_ref_results1)
        # psnr = no_ref_results[2]
        # if max_psnr < psnr:
        #     max_psnr = psnr
        #     list_max = file_name_i



        if fnb == num:

            print("------------------------------------------------ddddddd")
            #print(list_ref)
            #print(list_noref)
            temp_ref_results1, temp_no_ref_results1 = cal(list_ref, list_noref)
            ref_results.update({'deep   ':temp_ref_results1})
            no_ref_results.update({'deep    ':temp_no_ref_results1})


            print('################## reference comparision #######################')
            for index, i in enumerate(ref_results):
                if index == 0:
                    print(i, ref_results[i])
                else:
                    string_temp = [str(round(j, 4))+' &' for j in ref_results[i]]
                    string_temp = ' '.join(string_temp)
                    string_temp1 = 'deep    '+string_temp
                    print(string_temp1)
            print('################## reference comparision #######################')


            print('################## no reference comparision ####################')
            for index, i in enumerate(no_ref_results):
                if index == 0:
                    print(i, no_ref_results[i])
                else:
                    string_temp2 = [str(round(j, 4))+' &' for j in no_ref_results[i]]
                    string_temp2 = ' '.join(string_temp2)
                    string_temp3 = 'deep    '+string_temp2

                    # print(i, [round(j, 4) for j in no_ref_results[i]])
                    print(string_temp3)
            print('################## no reference comparision ####################')
            print(f'best file_name{list_max}')
            break


