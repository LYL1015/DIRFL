
import torch
from utils.helpers import  to_variable
from utils.SAM_loss import SAMLoss

def Compute_loss(config,out,pan,gt):
    """Set loss functions."""
    # criterion loss
    if config["loss"]["criterion"] == "L1":
        criterion   = torch.nn.L1Loss()
     
    elif config["loss"]["criterion"] == "None":
        print('self.loss_list[criterion] = None')
        criterion = None
    else:
        exit("Undefined criterion loss type")  
        
        
    """start compute loss"""
    # Normal L1 loss
    loss =0
    outputs = torch.clip(out['pred'],min=0,max=1)
    
    if criterion != None:
        if config["loss"]["Normalized_L1"]:
            max_ref     = torch.amax(gt, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(gt)
            loss        = criterion(outputs/max_ref, to_variable(gt)/max_ref)
        else:
            loss        = criterion(outputs, to_variable(gt))

    
    #  SAM_loss
    try :
        if config["loss"]["SAMLoss"]:
            loss += config["loss"]["SAMLoss_F"]*SAMLoss(outputs,to_variable(gt))
    except  KeyError:
        pass 

    #frequency loss
    try :
        if config["loss"]["frequency_loss"]:
            pred = torch.fft.rfft2(outputs+1e-8, norm='backward')
            gt = torch.fft.rfft2(gt+1e-8, norm='backward')
            loss_fn = torch.nn.L1Loss()
            pred_pha = torch.angle(pred)
            gt_pha = torch.angle(gt)
            loss1_fn  = loss_fn(pred,gt) + loss_fn(gt_pha,pred_pha)
            loss += config["loss"]["frequency_loss_F"]*loss1_fn
            
    except  KeyError:
        pass 
    
    
        
    return loss