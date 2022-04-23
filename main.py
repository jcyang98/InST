import argparse
import os
from tqdm import tqdm
import yaml
import logging
import sys
import datetime
from pathlib import Path
from PIL import Image
from tensorboardX import SummaryWriter
import time

import numpy as np
import torch
from torch.autograd import backward
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset import FlatFolderDataset,PairWiseDataset
# from models.model import AssembleNet

sys.path.append('mask_RAFT')

from mask_RAFT.mask_raft import mask_RAFT

from warp import apply_warp_by_field
from utils_save import save_loss,flow_to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser('Warp Model for fast warpping')
    # basic options
    parser.add_argument('--local_rank',type=int,default=-1,help='the rank of the process among all the processes of the local node')
    parser.add_argument('--cpu',action='store_true', help='wheather to use cpu , if not set, use gpu')
    parser.add_argument('--seed', type=int, default=42,help='random seed')

    # data options
    parser.add_argument('--source_dir',type=str,default='images/source_mask',help='Directory path to source images')
    parser.add_argument('--source2_dir',type=str,default=None,help='Directory path to source2 images')
    parser.add_argument('--target_dir',type=str,default='images/target_mask',help='Directory path to target images')
    parser.add_argument('--target2_dir',type=str,default=None,help='Directory path to target2 images')
    parser.add_argument('--im_height',type=int,default=256)
    parser.add_argument('--im_width',type=int,default=256)
    parser.add_argument('--num_worker',type=int,default=8)

    parser.add_argument('--pair_dir',type=str,default=None,help='Wheather to use pair dataset')
    parser.add_argument('--pair_txt',type=str,default=None,help='pair txt file')

    # model options
    parser.add_argument('--model_type',type=str,default='mask_raft',choices=['mask_raft'])
    parser.add_argument('--pretrained_model',type=str,help='module of warpping')
    parser.add_argument('--train_refine_time',type=int,default=3,help='warp refine time in training')
    parser.add_argument('--visual_refine_time',type=int,default=3,help='warp refine time in testing')
    parser.add_argument('--no_normalize_features',action='store_true',help='wheather to feature normalization')
    parser.add_argument('--no_normalize_matches',action='store_true',help='wheather to match normalization')

    # loss options
    parser.add_argument('--loss_method',type=str,default='mse',choices=['mse','classify'])
    parser.add_argument('--smooth_loss',type=str,default='1st',choices=['1st','2nd'])
    parser.add_argument('--smooth_mask',type=str,default='edge_xor',choices=['none','midgrid','xor','stylemask','or','gauss','style_edge','edge_xor'])
    parser.add_argument('--warp_weight',type=float,default=1)
    parser.add_argument('--reg_weight',type=float,default=0.1)
    parser.add_argument('--semantic_loss', action='store_true', help='add semantic loss, ignore it in our work')                    
    parser.add_argument('--sem_weight',type=float,default=1)
    parser.add_argument('--seq_gamma',type=float,default=0.1)

    # training options
    parser.add_argument('--num_iter',type=int,default=20000)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--optim',type=str,default='adam',choices=['sgd','adam','adamw'])
    parser.add_argument('--scheduler',type=str,default='StepLR',choices=['StepLR','OneCycleLR'])
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight_decay',type=float,default=5e-5)
    parser.add_argument('--lr_decay',type=float,default=0.999)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--sync_bn',action='store_true',help='use SyncBatchNorm, only available in DDP mode')

    # other options
    parser.add_argument('--saved_dir',type=str,default='./saved_dir')
    parser.add_argument('--save_checkpoint_interval',type=int,default=5000)
    parser.add_argument('--resume', type=str, help='resume training checkpoint')
    parser.add_argument('--write_loss_interval',type=int,default=100)

    # Visualize options
    parser.add_argument('--running_img_dir', type=str, default='images/running-images-mask')
    parser.add_argument('--no_visual',action='store_true',help='wheather to visualize')
    parser.add_argument('--visual_interval', type=int, default=200,
                        help="the interval of visualizing the result of the model")
    parser.add_argument('--make_grid', action='store_true', help='make gird of the test output')                    

    args = parser.parse_args()
    return args

def check_args(args):
    if args.cpu and args.local_rank != -1:
        logger.error('if you want to use cpu, do not use DDP')
        sys.exit()
    if not args.cpu and not torch.cuda.is_available():
        logger.error('cuda is not available, try running on CPU by adding param --cpu')
        sys.exit()
    
    # if args.semantic_loss:
    #     if args.source2_dir == None:
    #         logger.error('if you want use semantic loss on rgb images, you must provide source2_dir and target2_dir, otherwise just do not use semantic_loss')
    #         sys.exit()
    #     else:
    #         logger.info(f'[Info] You will use semantic loss on rgb images...')

    # Save Dir setting
    args.begin_time = datetime.datetime.now().strftime('%H%M%S')
    save_name = f'{args.model_type}_w{args.warp_weight}_r{args.reg_weight}_s{args.sem_weight}_lr{args.lr}_bs{args.batch_size}_refine{args.train_refine_time}' + \
                f'_ploss{args.loss_method}_sm{args.smooth_loss}_{"rgb_" if args.source2_dir != None else "mask_"}' + f'gamma{args.seq_gamma}_{args.smooth_mask}_' + args.begin_time
    args.saved_dir = Path(args.saved_dir) / save_name
    args.saved_dir.mkdir(exist_ok=True,parents=True)
    logger.info(f'Save Dir : {str(args.saved_dir)}')

    # Save options
    with open(args.saved_dir / 'opt.yaml','w') as f:
        yaml.dump(vars(args),f,sort_keys=False)


    if not args.no_visual:
        if not (Path(args.running_img_dir) / 'source').exists() or not (Path(args.running_img_dir) / 'target').exists():
            logger.error(f'you set the argument visual_interval , but if you want to visualize the results during training,'
                    f'you should create the folder {args.running_img_dir}/source which contains the source images and {args.running_img_dir}/target which contains target images')
            sys.exit(1)
        args.vis_output_dir = args.saved_dir / 'running-output'
        os.makedirs(args.vis_output_dir,exist_ok=True)

def _make_model(args,device):
    logger.info(f'Using {args.model_type} Model ......')
    logger.info(f'Use {args.smooth_loss} Smooth Loss')
    logger.info(f'Use {args.smooth_mask} Smooth Mask')
   
    if args.model_type == 'mask_raft':
        model = mask_RAFT(args.smooth_loss,args.smooth_mask,args.semantic_loss,args.seq_gamma).to(device)
        logger.info(f'seq_gamma : {args.seq_gamma}')

    model_parameters = filter(lambda p:p.requires_grad,model.parameters())
    n_params = sum([p.numel() for p in model_parameters])
    logger.info('Model Setting ...')
    logger.info(f'Number of params: {n_params}')

    if args.pretrained_model is not None:
        logger.info(f'Load Pretrained model from {args.pretrained_model}')
        model.load_state_dict(torch.load(args.pretrained_model)['modelstate'])

    # SyncBatchNorm
    if args.sync_bn and args.local_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info('Using SyncBatchNorm')
    
    if not args.cpu:
        if args.local_rank == -1 and torch.cuda.device_count() > 1:
            logger.warning('For multi gpus, you will use DataParallel. To spped up, you can try to use torch.distributed.launch for distribution')
            model = torch.nn.DataParallel(model)
        elif args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=[args.local_rank])
    return model

def _make_optimizer(args,model):
    logger.info(f'Using {args.optim} Optimizer ......')
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,eps=args.epsilon)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.99,nesterov=True,weight_decay=args.weight_decay)
    return optimizer

def _make_scheduler(args,optimizer):
    logger.info(f'Using {args.scheduler} Scheduler ......')
    if args.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma = args.lr_decay)
    elif args.scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_iter+100,
                        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return scheduler

def _get_transform(size=None):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def _make_data(args,type,shuffle=False):
    if type == 'pair':
        # dataset = PairWiseDataset(args.pair_dir,args.pair_txt,transform=_get_transform(size=(args.im_height,args.im_width)))
        dataset = PairWiseDataset(args.pair_dir,transform=_get_transform(size=(args.im_height,args.im_width)))
        logger.info(f'Number of pairs: {len(dataset)}')
    else:
        data_root = args.source_dir if type == 'source' else args.target_dir
        data2_root = args.source2_dir if type == 'source' else args.target2_dir
        if data2_root != None:
            logger.info(f'Use extra {type} root {data2_root}')
        dataset = FlatFolderDataset(data_root,data2_root,transform=_get_transform(size=(args.im_height,args.im_width)))
        logger.info(f'Number of {type} images: {len(dataset)}')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1 else None
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_worker,sampler=sampler,drop_last=True)
    return loader

def adjust_learning_rate(optimizer, lr, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+iteration*1e-5)

def intermediate_visual(model,args,device,batch_id):
    test_source_images = sorted(list((Path(args.running_img_dir)/'source').glob('*.jpg')) + list((Path(args.running_img_dir)/'source').glob('*.png')))
    test_target_images = sorted(list((Path(args.running_img_dir)/'target').glob('*.jpg')) + list((Path(args.running_img_dir)/'target').glob('*.png')))
    if len(test_source_images) == 0 or len(test_target_images) == 0: return
    
    rgb_flag = False
    if args.source2_dir == None and (Path(args.running_img_dir)/'source_rgb').exists():
        rgb_flag = True
        rgb_images = sorted(list((Path(args.running_img_dir)/'source_rgb').glob('*.jpg')) + list((Path(args.running_img_dir)/'source_rgb').glob('*.png')))
    checkerboard_flag = False
    if (Path(args.running_img_dir)/'checkerboard').exists():
        checkerboard_flag = True
        checkerboard_images = sorted(list((Path(args.running_img_dir)/'checkerboard').glob('*.jpg')) + list((Path(args.running_img_dir)/'checkerboard').glob('*.png')))

    output_dir = args.vis_output_dir / f'{batch_id}'
    output_dir.mkdir()
    source_test_transform= _get_transform(size=(args.im_height,args.im_width))
    target_test_transform = _get_transform(size=(args.im_height,args.im_width))

    if args.make_grid:
        c, h, w = target_test_transform(Image.open(test_target_images[0]).convert('RGB')).size()
        concat_list = []
        first_row = [torch.ones(1,c,h,w)]
        for target_path in test_target_images:
            target_img = Image.open(target_path).convert('RGB')
            target = target_test_transform(target_img).unsqueeze(0)
            first_row.append(target)
        first_row = torch.cat(first_row,dim=0)
        concat_list.append(first_row)

    for i,source_path in enumerate(test_source_images):
        source_img = Image.open(source_path).convert('RGB')
        source = source_test_transform(source_img).unsqueeze(0)
        if rgb_flag:
            rgb_source = Image.open(rgb_images[i]).convert('RGB')
            rgb_source = source_test_transform(rgb_source).unsqueeze(0).to(device)
        if checkerboard_flag:
            checkerboard_source = Image.open(checkerboard_images[i]).convert('RGB')
            checkerboard_source = source_test_transform(checkerboard_source).unsqueeze(0).to(device)
        if args.make_grid:
            row = [source]
        source = source.to(device)
        for target_path in test_target_images:
            target_img = Image.open(target_path).convert('RGB')
            target = target_test_transform(target_img).unsqueeze(0).to(device)
            with torch.no_grad():
                warp_fields,warped_source_list = model(source,target,refine_time=args.visual_refine_time,test=True)
                warped_source = warped_source_list[-1]

            if not args.make_grid:
                optical_flow_list = [torch.ones_like(source),torch.ones_like(source)]
                for warp_field in warp_fields:
                    optical_flow_np = flow_to_image(warp_field)     # [256,256,3] ndarray
                    optical_flow_tensor = torch.Tensor(optical_flow_np.astype(np.float) / 255).permute((2,0,1)).unsqueeze(0).to(device)
                    optical_flow_list.append(optical_flow_tensor)
                if rgb_flag:
                    extra_warped_source_list = [rgb_source,target]
                    for warp_field in warp_fields:
                        extra_warped_source = apply_warp_by_field(rgb_source.clone(), warp_field.clone(), device) 
                        extra_warped_source_list.append(extra_warped_source)   
                else:
                    extra_warped_source_list = []
                if checkerboard_flag:
                    warped_checkerboard_list = [checkerboard_source,target]
                    for warp_field in warp_fields:
                        warped_checkerboard = apply_warp_by_field(checkerboard_source.clone(), warp_field.clone(), device) 
                        warped_checkerboard_list.append(warped_checkerboard)
                else:
                    warped_checkerboard_list = [] 

            if args.make_grid:
                row.append(warped_source.cpu())
            else:
                save_image_name = Path(source_path).name.split('.')[0] + '_shaped_' + Path(target_path).name              
                save_image(torch.cat([source,target] + warped_source_list + extra_warped_source_list + warped_checkerboard_list + optical_flow_list,dim=0).cpu(),
                        str(Path(output_dir,save_image_name)),scale_each=True,nrow=2+args.visual_refine_time,padding=4,pad_value=255)

        if args.make_grid:
            concat_list.append(torch.cat(row,dim=0))
    if args.make_grid:
        concat = torch.cat(concat_list,dim=0)
        save_image(concat,Path(output_dir,'result.jpg'),scale_each=True,nrow=len(test_target_images)+1,padding=4,pad_value=255)

def train(args):
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        logger.info(f'process_{args.local_rank} starts training ...')
    device = torch.device('cuda' if not args.cpu else 'cpu')
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu:
        torch.cuda.manual_seed(args.seed)
    
    log_dir = args.saved_dir / 'tensorboard'
    log_dir.mkdir(exist_ok=True,parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Model =================================
    model = _make_model(args,device)

    # Optimizer =============================
    optimizer = _make_optimizer(args,model)

    # Scheduler =============================
    scheduler = _make_scheduler(args,optimizer)

    # Data ==================================
    if args.pair_dir is None:
        source_loader_ = _make_data(args,type='source')
        source_loader = iter(source_loader_)
        target_loader_ = _make_data(args,type='target')
        target_loader = iter(target_loader_)
    else:
        assert args.pair_dir is not None 
        logger.info(f'Using Pair Dataset ...')
        data_loader_ = _make_data(args,type='pair')
        data_loader = iter(data_loader_)

    iteration = -1
    # Resume
    if args.resume is not None:
        logger.info(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['modelstate'])
        optimizer.load_state_dict(checkpoint['optimstate'])
        iteration = checkpoint['iteration']

    ell_warp_TV_list = []
    ell_warp_list = []
    ell_warp_sem_list = []
    ell_list = []

    # Training ===============================
    model.train()
    try:
        progress_bar = tqdm(range(iteration+1,args.num_iter))
        for batch_id in progress_bar:
            source_2, target_2 = None,None
            if args.pair_dir is None:
                try:
                    source = source_loader.next()
                except StopIteration:
                    source_loader = iter(source_loader_)
                    source = source_loader.next()
                except:
                    continue

                try:
                    target = target_loader.next()
                except StopIteration:
                    target_loader = iter(target_loader_)
                    target = target_loader.next()
                except:
                    continue
                if isinstance(source,list): 
                    source,source_2 = source
                    source_2 = source_2.to(device)
                if isinstance(target,list): 
                    target,target_2 = target
                    target_2 = target_2.to(device)
            else:
                try:
                    source, target = data_loader.next()
                except StopIteration:
                    data_loader = iter(data_loader_)
                    source, target = data_loader.next()
                except:
                    continue
            
            _, _, ell_warp,ell_warp_TV,ell_warp_sem = model(source.to(device),target.to(device),refine_time=args.train_refine_time,image1_mask=source_2,image2_mask=target_2)
            ell = args.warp_weight * ell_warp + args.reg_weight * ell_warp_TV + args.sem_weight * ell_warp_sem

            # # Joint Training
            # source, target = source.to(device), target.to(device)
            # flow_predictions, _, ell_warp,ell_warp_TV,ell_warp_sem = model(source_2,target_2,refine_time=args.train_refine_time,image1_mask=None,image2_mask=None)   
            # warped_img1_rgb = apply_warp_by_field(source.clone(),flow_predictions[-1],device)
            # # (log_p, logdet, z_outs) = glow()
            # z_c = glow(warped_img1_rgb, forward=True)
            # z_s = glow(target, forward=True)
            # # reverse 
            # stylized = glow(z_c, forward=False, style=z_s)
            # loss_c, loss_s = encoder(warped_img1_rgb, target, stylized)
            # loss_c = loss_c.mean()
            # loss_s = loss_s.mean()
            # ell = args.warp_weight * ell_warp + args.reg_weight * ell_warp_TV + args.sem_weight * ell_warp_sem + 0.01*loss_c + 0.01*loss_s      
            

            if args.local_rank in [-1,0]:
                progress_bar.set_description(
                'Iteration: {}/{} warp_loss: {:.5f} reg_loss: {:.5f} sem_loss: {:.5f} Loss: {:.5f} lr: {:.4f}'.format(
                    batch_id,args.num_iter,ell_warp.item(),ell_warp_TV.item(),ell_warp_sem.item(),ell.item(),optimizer.state_dict()['param_groups'][0]['lr']))
                
                ell_warp_TV_list.append(ell_warp_TV.item())
                ell_warp_list.append(ell_warp.item())
                ell_warp_sem_list.append(ell_warp_sem.item())
                ell_list.append(ell.item())
                writer.add_scalars('Loss', {'Total Loss': ell.item(),'Warp Loss': ell_warp.item(),'Warp_TV Loss': ell_warp_TV.item(),'Warp_Sem Loss': ell_warp_sem.item()}, batch_id)

                # Intermediate Visualization
                if not args.no_visual and batch_id % args.visual_interval == 0:
                    logger.info('Running Time Visualization ...')
                    model.eval()
                    intermediate_visual(model,args,device,batch_id)
                    model.train()

                if batch_id % args.save_checkpoint_interval == 0 and batch_id != 0:
                    checkpoint_path = args.saved_dir / f'{batch_id}.pth'
                    save_checkpoint(checkpoint_path,model,optimizer,batch_id)
                    logger.info(f'Intermediate save, Model saved at {str(checkpoint_path)}')
                
                if batch_id % args.write_loss_interval == 0:
                    save_loss(str(args.saved_dir),ell_warp_list,ell_warp_TV_list,ell_warp_sem_list,ell_list,args.warp_weight,args.reg_weight,args.sem_weight)
            
            optimizer.zero_grad()
            ell.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            # adjust_learning_rate(optimizer,args.lr,batch_id)
            # if batch_id / 50 <= 200:
            #     scheduler.step()
            scheduler.step()
    except KeyboardInterrupt:
        logger.info('Catch a KeyboardInterupt')
    
    if args.local_rank in [-1,0]:
        checkpoint_path = args.saved_dir / f'{batch_id}.pth'
        save_checkpoint(checkpoint_path,model,optimizer,batch_id)
        logger.info(f'Training Done, Model saved at {str(checkpoint_path)}')

def save_checkpoint(checkpoint_path,model,optimizer,batch_id):
    if isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint = {
        'modelstate':model_state_dict,
        'optimstate':optimizer.state_dict(),
        'iteration':batch_id,
        }
    torch.save(checkpoint,checkpoint_path)
    

def main():
    args = get_args()
    print(args)
    if args.local_rank in [-1,0]:
        check_args(args)
    train(args)

if __name__ == '__main__':
    main()

