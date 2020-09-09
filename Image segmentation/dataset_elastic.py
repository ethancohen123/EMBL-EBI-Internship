# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:34:08 2020

@author: ethan
"""


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import numpy as np
from util import load_set



from albumentations import (
    Resize,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)


original_height=512
original_width=512
transform_train=Compose([
    VerticalFlip(p=0.5),              
    RandomRotate90(p=0.5),
    OneOf([
        ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.2),
        OpticalDistortion(p=0.4, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8)
         ])
transform_val=Compose([
   Resize(512,512)])
transform_nolastic=Compose([
    VerticalFlip(p=0.5),              
    RandomRotate90(p=0.5)])



class Dataloader(Dataset):
    def __init__(self,img_fol,mask_fol,transform=None):
        self.img_fol=img_fol
        self.mask_fol=mask_fol
        self.transform=transform
    def __getitem__(self, idx):
        image=load_set(self.img_fol,is_mask=False)[0][idx]
        mask=load_set(self.mask_fol,is_mask=True)[0][idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image=augmented['image']
            mask=augmented['mask']
            normalize_img= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            image=normalize_img(image)
            image=image.permute(0,2,1) 
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            mask=transform_to_tensor(mask)


        else:
            normalize_img= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            image=normalize_img(image)
            image=image.permute(0,2,1) 
            transform_to_tensor = transforms.Compose([transforms.ToTensor()]) 
            mask=transform_to_tensor(mask)                        
        
        return image, mask
 

    def __len__(self):
        return len(self.mask_fol)
    
    
    
def get_loader(path_img,path_mask, validation_split=.20,  shuffle_dataset=True):
    dataset = Dataloader(path_img,path_mask)  # instantiating the data set.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split_val = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices = indices[split_val :]
    val_indices = indices[: split_val]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    
    dataset_train=Dataloader(path_img,path_mask,transform=transform_train)
    dataset_val=Dataloader(path_img,path_mask,transform=transform_val)
    loader = {
        'train': DataLoader(dataset_train, batch_size=4, sampler=train_sampler),
        'val': DataLoader(dataset_val, batch_size=1, sampler=valid_sampler),
    }
    return loader

