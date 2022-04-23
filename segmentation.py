from numpy.core.function_base import add_newdoc
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

def segment(data_paths):
    print('total images number: ',len(data_paths))

    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    # cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

    cfg.MODEL.WEIGHTS = "detectron2_repo/model_final_edd263.pkl"

    predictor = DefaultPredictor(cfg)

    added_mask_num = 0
    mask_paths = []
    for data_path in tqdm(data_paths):
        try:
            im = cv2.imread(str(data_path))
            outputs = predictor(im)
            instance_num = outputs['instances'].pred_masks.shape[0]
            if instance_num == 0:
                print(f'images {data_path} has no instances')
                # mask = torch.zeros(im.shape[0],im.shape[1])
                # max_ins_id = -1
                exit()
            else:
                max_ins_id = 0
                bbox = outputs['instances'].pred_boxes[0].tensor.squeeze()
                max_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                for i in range(1,instance_num):
                    bbox = outputs['instances'].pred_boxes[i].tensor.squeeze()
                    area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                    if area > max_area:
                        max_ins_id = i
                mask = outputs['instances'].pred_masks[max_ins_id].float()
            
            mask = mask.unsqueeze(dim=0).unsqueeze(dim=0).cpu()
            mask_path = data_path.parent/f'{data_path.stem}_mask.jpg'
            save_image(mask,mask_path)
            mask_paths.append(mask_path)
            added_mask_num += 1

            # if max_ins_id != -1:
            #     rect = outputs['instances'].pred_boxes[max_ins_id].tensor.int().squeeze().tolist()
            # else:
            #     rect = None
            # mask = mask.cpu().numpy().astype('uint8')
            # # mask = np.zeros(im.shape[:2],np.uint8)
            # bgdModel = np.zeros((1,65),np.float64)
            # fgdModel = np.zeros((1,65),np.float64)
            # cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
            # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            # save_image(torch.from_numpy(mask2).float().unsqueeze(dim=0).unsqueeze(dim=0).cpu(),os.path.join(args.output_dir,os.path.basename(data_path)))
        except KeyboardInterrupt:
            break
        except BaseException as e:
            print(e)
            print('except_path: ',data_path)
            continue

    print('added mask number: ',added_mask_num)
    return mask_paths