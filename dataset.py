from pathlib import Path
from numpy.core.fromnumeric import sort
from torch.utils.data import Dataset,DataLoader, dataloader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time
import random
import os 
import glob

class PairWiseDataset(Dataset):
    def __init__(self, data_root, pair_txt=None, transform=None):
        self.data_root = data_root
        self.path_txt = pair_txt
        if pair_txt != None:
            self.pair_list = []
            with open(pair_txt,'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    pair = line.split()
                    self.pair_list.append([pair[0],pair[1]])
        else:
            self.transform2 = transforms.Compose([transform.transforms[0],transforms.Resize(128),transforms.RandomRotation(90,fill=(255,255,255)),transforms.Pad((256-128)//2,fill=(255,255,255)),transform.transforms[1]])
            self.paths = sorted(glob.glob(os.path.join(data_root,'*.jpg')) + glob.glob(os.path.join(data_root,'*.png')))
        self.transform = transform

    def __getitem__(self,index):
        if self.path_txt != None:
            pair = self.pair_list[index]
            img1 = Image.open(os.path.join(self.data_root,pair[0])).convert('RGB')
            img2 = Image.open(os.path.join(self.data_root,pair[1])).convert('RGB')
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        else:
            path = self.paths[index]
            img1_pil = Image.open(path).convert('RGB')
            if self.transform:
                img1 = self.transform(img1_pil)
                img2 = self.transform2(img1_pil)
        return img1,img2

    def __len__(self):
        if self.path_txt != None:
            return len(self.pair_list)
        else:
            return len(self.paths)

def PF_Pascal_txt(data_root,txt_dir):
    os.makedirs(txt_dir,exist_ok=True)
    l_cls = os.listdir(data_root)

    total_f = open(os.path.join(txt_dir,'total_pair'+'.txt'),'w')

    for cls in l_cls:
        image_root = os.path.join(data_root,cls)  

        if not os.path.isdir(image_root):
            continue
        txt_path = os.path.join(txt_dir,cls+'.txt')
    
        pair_names = os.listdir(image_root)
        
        with open(txt_path,'w') as f:
            for pair in pair_names:
                pair_1 = pair.split('-')[0] + '.jpg'
                pair_2 = pair.split('-')[1]
                f.write(pair_1 + ' ' + pair_2 + '\n')
                total_f.write(pair_1 + ' ' + pair_2 + '\n')
    total_f.close()
    
class FlatFolderDataset(Dataset):
    def __init__(self,data1_root,data2_root=None,transform=None):
        super().__init__()
        self.data1_root = data1_root
        self.data2_root = data2_root
        self.paths_1 = sorted(glob.glob(os.path.join(data1_root,'*.jpg')) + glob.glob(os.path.join(data1_root,'*.png')))
        if self.data2_root != None:
            self.paths_2 = sorted(glob.glob(os.path.join(data2_root,'*.jpg')) + glob.glob(os.path.join(data2_root,'*.png')))
            assert len(self.paths_1) == len(self.paths_2)
        self.transform = transform
    
    def __getitem__(self,index):
        path_1 = self.paths_1[index]
        img_1 = Image.open(path_1).convert('RGB')
        if self.transform:
            img_1 = self.transform(img_1)      
        if self.data2_root == None:
            return img_1

        path_2 = self.paths_2[index]
        assert os.path.basename(path_1) == os.path.basename(path_2)
        img_2 = Image.open(path_2).convert('RGB')
        if self.transform:  
            img_2 = self.transform(img_2)
        return img_1, img_2
    
    def __len__(self):
        return len(self.paths_1)

    def name(self):
        return 'FlatFolderDataset'

# class FlatFolderDataset(Dataset):
#     def __init__(self, root, transform=None,set_shuffle=False):
#         super(FlatFolderDataset, self).__init__()
#         self.root = root
#         self.paths = sorted(list(Path(self.root).glob('*')))
#         if set_shuffle:
#             random.shuffle(self.paths)
#         self.transform = transform

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(str(path)).convert('RGB')
#         # img = Image.open(str(path))
#         if self.transform:
#             img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.paths)

#     def name(self):
#         return 'FlatFolderDataset'

def _get_transform(size=None,crop=None):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    if crop is not None:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
    