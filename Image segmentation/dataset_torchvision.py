# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:26:59 2020

@author: ethan
"""

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch
import os
import numpy as np
import glob
from PIL import Image
from util import load_set



padding = transforms.Compose([transforms.Pad(20, padding_mode='reflect'),
                              transforms.RandomRotation((-6, 6)),
                              transforms.RandomApply([transforms.RandomAffine(0, shear=6)]),
                              transforms.RandomCrop(128)])

rescaling = transforms.Compose([transforms.Resize(128),
                                transforms.RandomApply([transforms.RandomAffine(0, shear=6)]),
                                transforms.RandomRotation((-6, 6))])

crop_rescaling = transforms.Compose([transforms.RandomCrop(84),
                                     transforms.Resize(128),
                                     transforms.RandomRotation((-6, 6))])


transform_train = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.RandomChoice([padding, rescaling, crop_rescaling]),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomApply([transforms.ColorJitter(brightness=0.1,
                                                                                                   contrast=0.1,
                                                                                                   saturation=0.1,
                                                                                                   hue=0.1)]),
                                                    transforms.ToTensor()
                                                    ])



transform_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(128),
                                    transforms.ToTensor()])



class Embryo(Dataset):
    def __init__(self,img_fol,mask_fol,transform=None):
        self.img_fol=img_fol
        self.mask_fol=mask_fol
        self.transform=transform
    def __getitem__(self, idx):
        img=load_set(self.img_fol,is_mask=False)[0][idx]
        mask=load_set(self.mask_fol,is_mask=True)[0][idx]
        #img_name=load_set(self.img_fol,is_mask=False)[1][idx] just to check if loaded image and mask the same
        #mask_name=load_set(self.mask_fol,is_mask=True)[1][idx]
        if self.transform:
            img = self.transform(img)
            normalize_img= transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            img=normalize_img(img)
            img=img.permute(0,2,1) 
            #transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            #mask=transform_to_tensor(mask)
            mask=self.transform(mask)
                              
        
        else:
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            mask=transform_to_tensor(mask)
            #mask=mask.permute(0,2,1)
            '''
            transform_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)
                #Normalizing makes images appear weird but necessary for resnet
            ])
            img = transform_to_tensor(img)
            '''
            img = transform_to_tensor(img)
            img=img.permute(0,2,1)                          
        
        return img, mask#,img_name,mask_name
 

    def __len__(self):
        return len(self.mask_fol)
    
    
    
def get_emb_loader(path_img,path_mask, validation_split=.20, test_split=.10, shuffle_dataset=True):
    dataset = Embryo(path_img,path_mask)  # instantiating the data set.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split_val = int(np.floor(validation_split * dataset_size))
    split_test = int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices = indices[split_val + split_test:]
    val_indices = indices[split_test:split_test + split_val]
    test_indices = indices[:split_test]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    dataset_train=Embryo(path_img,path_mask,transform=transform_train)
    dataset_val=Embryo(path_img,path_mask,transform=transform_val)
    loader = {
        'train': DataLoader(dataset_train, batch_size=1, sampler=train_sampler),
        'val': DataLoader(dataset_val, batch_size=1, sampler=valid_sampler),
        'test': DataLoader(dataset_val, batch_size=1, sampler=test_sampler)
    }
    return loader   

