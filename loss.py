
# Helper functions for calculating loss

# Code based on https://github.com/futscdav/strotss

import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image


def sequence_L1_p_loss(warped_I_c_list,I_g,gamma=0.8):
    n_predictions = len(warped_I_c_list)    
    ell_warp = 0.0
    for i in range(n_predictions):
        # i_weight = gamma**(n_predictions - i - 1)

        ## refine 4 
        # i_weight = [1/8,1/4,1/2,1][i]
        
        # refine 3
        # i_weight = [1/8,1/4,1][i]
        i_weight = [1/10,1/5,1][i]

        i_loss = (warped_I_c_list[i] - I_g).abs()
        # ell_warp += i_weight * i_loss.mean()
        ell_warp = ell_warp + i_weight * i_loss.mean()
    return ell_warp

def TV_with_mask(x,mask,alpha=0.9999,mask_type='binary'):
    # alpha1 = alpha
    # alpha2 = 1-alpha1
    assert mask_type in ['binary','weight']
    if mask_type == 'binary':
        mask = mask.bool()
        mask = torch.where(mask,torch.ones_like(mask)*alpha,torch.ones_like(mask)*(1-alpha))
    n = x.shape[2] * x.shape[3]
    ori_sum = mask.flatten(2).sum(dim=2).view(mask.shape[0],1,1,1) 
    mask = (n / ori_sum) * mask
    t = torch.pow(torch.abs(x[:,:,1:,: ] - x[:,:,0:-1,:  ]), 2)
    ell = (t * mask[:,:,1:,:]).mean()
    t = torch.pow(torch.abs(x[:,:,: ,1:] - x[:,:,:  ,0:-1]), 2)
    ell += (t * mask[:,:,: ,1:]).mean()
    t = torch.pow(torch.abs(x[:,:,1:,1:] - x[:,:, :-1, :-1]), 2)
    ell += (t * mask[:,:,1:,1:]).mean()
    t = torch.pow(torch.abs(x[:,:,1:,:-1] - x[:,:,:-1,1:]), 2)
    ell += (t * mask[:,:,1:,:-1]).mean()
    ell /= 4.
    return ell

def sequence_TV_with_mask_loss(warp_fields,mask,alpha=0.9999,gamma=0.1):
    n_predictions = len(warp_fields)    
    ell_warp_TV = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        ell_warp_TV += i_weight * TV_with_mask(warp_fields[i],mask,alpha)
    return ell_warp_TV

def TV(x):
    ell =  torch.pow(torch.abs(x[:,:,1:,: ] - x[:,:,0:-1,:  ]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,: ,1:] - x[:,:,:  ,0:-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,1:] - x[:,:, :-1, :-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,:-1] - x[:,:,:-1,1:]), 2).mean()
    ell /= 4.
    return ell

def sequence_TV_loss(warp_fields,gamma=0.1):
    n_predictions = len(warp_fields)    
    ell_warp_TV = 0.0
    for i in range(n_predictions):
        # i_weight = gamma**(n_predictions - i - 1)
        
        ## refine 4 
        # i_weight = [0.08,0.04,0.02,0.01][i]

        # refine 3
        # i_weight = [0.08,0.04,0.01][i]
        i_weight = [0.1,0.05,0.01][i]


        # ell_warp_TV += i_weight * TV(warp_fields[i])
        ell_warp_TV = ell_warp_TV + i_weight * TV(warp_fields[i])
    return ell_warp_TV

def gradients(img):
    dy = img[:,:,1:,:] - img[:,:,:-1,:]
    dx = img[:,:,:,1:] - img[:,:,:,:-1]
    return dx, dy

def cal_grad2_error(flow, img):
    img_grad_x, img_grad_y = gradients(img)
    w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
    w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

    dx, dy = gradients(flow)
    dx2, _ = gradients(dx)
    _, dy2 = gradients(dy)
    error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
    #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
    return error / 2.0

'''
Reference: Occulsion Aware Unsupervised Learning of Optical Flow from Video
https://github.com/jianfenglihg/UnOpticalFlow/tree/dc1a4af33b99c8e47b751f6dbec4140b67b099c3
'''
def sequence_edge_aware_2nd_loss(warp_fields,I_g,gamma=0.8):
    n_predictions = len(warp_fields)    
    ell_warp_TV = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        warp_field = warp_fields[i]
        ell_warp_TV += i_weight * cal_grad2_error(warp_field/20.0,I_g)
    ell_warp_TV = ell_warp_TV.mean()
    return ell_warp_TV


def second_order_smooth_loss(x):
    # delta_u, delta_v, mask = _second_order_deltas(flow)
    ell =  torch.pow(torch.abs(x[:,:,2:,:] + x[:,:,:-2,:] - 2*x[:,:,1:-1,:]), 2).mean()
    ell +=  torch.pow(torch.abs(x[:,:,:,2:]  + x[:,:,:,:-2] - 2*x[:,:,:,1:-1]), 2).mean()
    ell +=  torch.pow(torch.abs(x[:,:,2:,2:] + x[:,:,:-2,:-2] - 2*x[:,:,1:-1,1:-1]), 2).mean()
    ell +=  torch.pow(torch.abs(x[:,:,:-2,2:] + x[:,:,2:,:-2] - 2*x[:,:,1:-1,1:-1]), 2).mean()
    ell /= 4.
    return ell


    