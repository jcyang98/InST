
'''
Reference: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
Code based on https://github.com/princeton-vl/RAFT/tree/master/core
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os 

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

from loss import sequence_L1_p_loss,sequence_TV_loss,sequence_edge_aware_2nd_loss,sequence_TV_with_mask_loss
from warp import apply_warp_by_field
   
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def _get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class mask_RAFT(nn.Module):
    def __init__(self,smooth_loss='2nd',smooth_mask='none',semantic_loss=False,seq_gamma=0.1):
        super(mask_RAFT, self).__init__()
        class Args(object):
            def __init__(self):
                self.small = False
                self.alternate_corr = False
                self.dropout = 0
                self.mixed_precision = False
                self.gamma = seq_gamma
                
        args = Args()
        self.args = args
        self.smooth_loss = smooth_loss
        self.smooth_mask = smooth_mask
        self.semantic_loss = semantic_loss

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)   
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        # self.pos_embed = _get_sinusoid_encoding_table(32*32,256).permute(0,2,1).view(1,-1,32,32)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, refine_time=12, flow_init=None, upsample=True, test=False,image1_mask=None,image2_mask=None,corr_map=None):

        if image1_mask == None:
            image1_mask = image1
        if image2_mask == None:
            image2_mask = image2

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])   
            # fmap1, fmap2 = self.fnet([pos_image1, pos_image2])        
 
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # Position Embedding
        pos_embed = _get_sinusoid_encoding_table(fmap1.shape[2]*fmap1.shape[3],fmap1.shape[1]).permute(0,2,1).view(1,-1,fmap1.shape[2],fmap1.shape[3])
        fmap1 = fmap1 + pos_embed.to(fmap1.device)
        fmap2 = fmap2 + pos_embed.to(fmap2.device)

        # fmap1 = fmap1 + self.pos_embed_learn
        # fmap2 = fmap2 + self.pos_embed_learn

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            # corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius,q=self.q,k=self.k)
        
        if isinstance(corr_map,list):
            corr_map[0] = corr_fn.corr_map

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = self.cnet(image2)

            cnet = fmap2   
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        # down_flow_predictions = []
        for itr in range(refine_time):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
            # down_flow_predictions.append(coords1 - coords0)

        warped_img1_list = []
        for flow_up in flow_predictions:
            warped_img1 = apply_warp_by_field(image1_mask.clone(),flow_up,flow_up.device)
            warped_img1_list.append(warped_img1)

        if test:
            # return coords1 - coords0, flow_up
            return flow_predictions, warped_img1_list

        ell_warp = sequence_L1_p_loss(warped_img1_list,image2_mask,gamma=self.args.gamma)

        # In our work, you can ignore this semantic loss
        ell_warp_sem = torch.zeros_like(ell_warp)
        if self.semantic_loss and image1_mask != None and image2_mask != None:
            warped_img1_rgb = apply_warp_by_field(image1.clone(),flow_predictions[-1],flow_predictions[-1].device)
            feat_warped_img1 = self.feat_extra(gray(warped_img1_rgb)).pool4
            feat_img2 = self.feat_extra(gray(image2)).pool4
            ell_warp_sem += sequence_L1_p_loss([feat_warped_img1],feat_img2)

        if self.smooth_loss == '1st':
            if self.smooth_mask == 'none':
                ell_warp_TV = sequence_TV_loss(flow_predictions,self.args.gamma)
            else:
                content_mask = torch.where(image1_mask>0.5,torch.ones_like(image1_mask),torch.zeros_like(image1_mask))
                content_mask = (content_mask[:,0,:,:] * content_mask[:,1,:,:] * content_mask[:,2,:,:]).unsqueeze(dim=1).bool()    # [N,1,H,W]
                style_mask = torch.where(image2_mask>0.5,torch.ones_like(image2_mask),torch.zeros_like(image2_mask))
                style_mask = (style_mask[:,0,:,:] * style_mask[:,1,:,:] * style_mask[:,2,:,:]).unsqueeze(dim=1).bool()    # [N,1,H,W]
                c_share = content_mask & style_mask    # [N,1,H,W]
                c_xor = (content_mask ^ style_mask) & style_mask    # [N,1,H,W]
                # if self.smooth_mask == 'midgrid':
                #     # sm_mask = c_xor    # [N,1,H,W]
                #     grid_size = 8
                #     checkerboard_mask = gen_checkerboard(image1.shape[2],image1.shape[3],grid_size).to(image1.device) 
                #     sm_mask = checkerboard_mask & c_share | c_xor
                if self.smooth_mask == 'xor':
                    sm_mask = c_xor    # [N,1,H,W]
                elif self.smooth_mask == 'stylemask':
                    sm_mask = style_mask
                elif self.smooth_mask == 'or':
                    sm_mask = style_mask | content_mask
                elif self.smooth_mask == 'style_edge':
                    f = 9
                    assert f % 2 == 1
                    kernel = torch.ones(1,1,f,f)
                    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
                    sm_mask = F.conv2d(style_mask.float().cpu(), kernel, padding=(f-1)//2)
                    sm_mask = ((sm_mask > 0) & (sm_mask < f*f)).to(image1.device) 
                elif self.smooth_mask == 'edge_xor':
                    f = 9
                    assert f % 2 == 1
                    kernel = torch.ones(1,1,f,f)
                    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
                    sm_mask = F.conv2d(style_mask.float().cpu(), kernel, padding=(f-1)//2)
                    sm_mask = ((sm_mask > 0) & (sm_mask < f*f)).to(image1.device) 
                    sm_mask = (sm_mask & content_mask) | c_xor 
                # save_image(sm_mask.float().cpu(),'sm_mask_raft.jpg',scale_each=True,nrow=image1.shape[0],padding=0,pad_value=255)        
                ell_warp_TV = sequence_TV_with_mask_loss(flow_predictions,sm_mask,0.9999,self.args.gamma)
        elif self.smooth_loss == '2nd':
            ell_warp_TV = sequence_edge_aware_2nd_loss(flow_predictions,image2_mask,self.args.gamma)
            # ell_warp_TV = 0.5*sequence_edge_aware_2nd_loss(flow_predictions,image2_mask,self.args.gamma) + 0.5*sequence_TV_loss(flow_predictions,self.args.gamma)
        
        return flow_predictions, warped_img1_list, ell_warp, ell_warp_TV, ell_warp_sem
