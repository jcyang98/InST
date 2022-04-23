import argparse
import os
import logging
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append('mask_RAFT')
from mask_RAFT.mask_raft import mask_RAFT

from warp import apply_warp_by_field
from utils_save import flow_to_image
from segmentation import segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser('Test Warp Model for fast warpping')
    # basic options
    parser.add_argument('--cpu',action='store_true', help='wheather to use cpu , if not set, use gpu')

    # data options
    parser.add_argument('--source_path',type=str,help='path of source image')
    parser.add_argument('--source_dir',type=str,help='Directory path to source image')
    parser.add_argument('--source_mask_dir',type=str,help='Directory path to source mask images, if None, use PointRender to segment')
    parser.add_argument('--source_mask_path',type=str,help='path of source mask image, if None, use PointRender to segment')
    parser.add_argument('--checkerboard_path',type=str,help='path of checkerboard image')
    parser.add_argument('--checkerboard_dir',type=str,help='Directory path to checkerboard image')
    parser.add_argument('--target_dir',type=str,help='Directory path to target images')
    parser.add_argument('--target_path',type=str,help='path of target images')
    parser.add_argument('--target_mask_dir',type=str,help='Directory path to target mask images, if None, use PointRender to segment')
    parser.add_argument('--target_mask_path',type=str,help='path of target mask images, if None, use PointRender to segment')
    parser.add_argument('--im_height',type=int,default=256)
    parser.add_argument('--im_width',type=int,default=256)
    parser.add_argument('--num_worker',type=int,default=8)

    # model options
    parser.add_argument('--model_type',type=str,default='mask_raft',choices=['mask_raft'],help='module of warpping')
    parser.add_argument('--checkpoint',type=str,required=True,help='module of warpping')
    parser.add_argument('--refine_time',type=int,default=3,help='warp refine time')

    # other options
    parser.add_argument('--output_dir',type=str,default='./output')
    parser.add_argument('--make_grid', action='store_true', help='make gird of the test output')                    

    args = parser.parse_args()
    return args

def _get_transform(size=None):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform    

def test(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    
    extra_flag = False
    if args.source_path or args.source_dir:
        extra_flag = True
        if args.source_path:
            source_paths = [Path(args.source_path)]
        else:
            source_dir = Path(args.source_dir)
            source_paths = sorted([f for f in source_dir.glob('*.jpg')] + [f for f in source_dir.glob('*.png')] + [f for f in source_dir.glob('*.jpeg')])

    if args.source_mask_path:
        source_mask_paths = [Path(args.source_mask_path)]
    elif args.source_mask_dir:
        source_mask_dir = Path(args.source_mask_dir)
        source_mask_paths = sorted([f for f in source_mask_dir.glob('*.jpg')] + [f for f in source_mask_dir.glob('*.png')] + [f for f in source_mask_dir.glob('*.jpeg')])
    else:
        # Use PointRend to help segment
        assert extra_flag
        logger.info('Using PointRender to segment source images. This will take a while, if you want to save time and generate more accurate results,'
                    'you can provide source mask and target mask path in the command line')
        source_mask_paths = segment(source_paths)


    if args.target_path or args.target_dir:
        if args.target_path:
            target_paths = [Path(args.target_path)]
        else:
            target_dir = Path(args.target_dir)
            target_paths = sorted([f for f in target_dir.glob('*.jpg')] + [f for f in target_dir.glob('*.png')] + [f for f in target_dir.glob('*.jpeg')])

    if args.target_mask_path:
        target_mask_paths = [Path(args.target_mask_path)]
    elif args.target_mask_dir:
        target_mask_dir = Path(args.target_mask_dir)
        target_mask_paths = sorted([f for f in target_mask_dir.glob('*.jpg')] + [f for f in target_mask_dir.glob('*.png')] + [f for f in target_mask_dir.glob('*.jpeg')])
    else:
        assert args.target_path or args.target_dir
        logger.info('Using PointRender to segment target images...')
        target_mask_paths = segment(target_paths)


    checkerboard_flag = False
    if args.checkerboard_path or args.checkerboard_dir:
        checkerboard_flag = True
        if args.checkerboard_path:
            assert args.source_path
            checkerboard_paths = [Path(args.checkerboard_path)]
        else:
            assert args.source_dir
            checkerboard_dir = Path(args.checkerboard_dir)
            checkerboard_paths = sorted([f for f in checkerboard_dir.glob('*.jpg')] + [f for f in checkerboard_dir.glob('*.png')] + [f for f in checkerboard_dir.glob('*.jpeg')])

    if len(source_mask_paths) == 0 or len(target_mask_paths) == 0: 
        logger.warn('no test images')
        return

    # Model
    logger.info(f'Using {args.model_type} for Testing')
    if args.model_type == 'mask_raft':
        model = mask_RAFT().to(device)

    checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    model.load_state_dict(checkpoint['modelstate'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    source_transform = _get_transform(size=(args.im_height,args.im_width))
    target_transform = _get_transform(size=(args.im_height,args.im_width))

    if args.make_grid:
        c, h, w = target_transform(Image.open(target_mask_paths[0]).convert('RGB')).size()
        concat_list = []
        first_row = [torch.ones(1,c,h,w)]
        for target_path in target_mask_paths:
            target_img = Image.open(target_path).convert('RGB')
            target = target_transform(target_img).unsqueeze(0)
            first_row.append(target)
        first_row = torch.cat(first_row,dim=0)
        concat_list.append(first_row)

    model.eval()
    for i,source_mask_path in enumerate(source_mask_paths):
        source_mask_img = Image.open(source_mask_path).convert('RGB')
        source_mask = source_transform(source_mask_img).unsqueeze(0)

        if extra_flag:
            source_img = Image.open(source_paths[i]).convert('RGB')
            source = source_transform(source_img).unsqueeze(0).to(device)
        if checkerboard_flag:
            checkerboard_img = Image.open(checkerboard_paths[i]).convert('RGB')
            checkerboard_source = source_transform(checkerboard_img).unsqueeze(0).to(device)

        if args.make_grid:
            row = [source_mask]
        source_mask = source_mask.to(device)
        for target_mask_path in target_mask_paths:
            target_mask_img = Image.open(target_mask_path).convert('RGB')
            target_mask = target_transform(target_mask_img).unsqueeze(0).to(device)
            corr_map = [None]
            with torch.no_grad():
                warp_fields,warped_source_mask_list = model(source_mask,target_mask,refine_time=args.refine_time,test=True,corr_map=corr_map)
                warped_source = warped_source_mask_list[-1] 
            corr_map = torch.mean(corr_map[0],dim=1).unsqueeze(1)
            if not args.make_grid:
                # Transform optical flow tensor
                optical_flow_list = [torch.ones_like(source),torch.ones_like(source)]
                for warp_field in warp_fields:
                    optical_flow_np = flow_to_image(warp_field)     # [256,256,3] ndarray
                    optical_flow_tensor = torch.Tensor(optical_flow_np.astype(np.float) / 255).permute((2,0,1)).unsqueeze(0).to(device)
                    optical_flow_list.append(optical_flow_tensor)
                # Extra source images warp
                if extra_flag:
                    warped_source_list = [source,target_mask]
                    for warp_field in warp_fields:
                        warped_source = apply_warp_by_field(source.clone(), warp_field.clone(), device) 
                        warped_source_list.append(warped_source)
                else:
                    warped_source_list = []
                if checkerboard_flag:
                    warped_checkerboard_list = [checkerboard_source,target_mask]
                    for warp_field in warp_fields:
                        warped_checkerboard = apply_warp_by_field(checkerboard_source.clone(), warp_field.clone(), device) 
                        warped_checkerboard_list.append(warped_checkerboard)
                else:
                    warped_checkerboard_list = []

            if args.make_grid:
                row.append(warped_source.cpu())
            else:
                save_image_name = source_mask_path.name.split('.')[0] + '_shaped_' + target_mask_path.name.split('.')[0] + f'_refine{args.refine_time}.jpg'   
                save_image(torch.cat([source_mask,target_mask] + warped_source_mask_list + warped_source_list + warped_checkerboard_list + optical_flow_list,dim=0).cpu(),
                                str(Path(output_dir,save_image_name)),scale_each=True,nrow=2+args.refine_time,padding=4,pad_value=255)  
                if extra_flag:
                    single_image_name = f'single_' + source_mask_path.name.split('.')[0] + '_shaped_' + target_mask_path.name.split('.')[0] + f'_refine{args.refine_time}.jpg'   
                    save_image(warped_source_list[-1].cpu(),str(Path(output_dir,single_image_name)),scale_each=True,nrow=1,padding=4,pad_value=255)

        if args.make_grid:
            concat_list.append(torch.cat(row,dim=0))
    if args.make_grid:
        concat = torch.cat(concat_list,dim=0)
        save_image(concat,Path(output_dir,'grid_result.jpg'),scale_each=True,nrow=len(target_mask_paths)+1,padding=4,pad_value=255)


if __name__ == '__main__':
    args = get_args()
    test(args)

