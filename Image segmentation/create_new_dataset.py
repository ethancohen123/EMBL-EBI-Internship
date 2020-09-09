# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:35:18 2020

@author: ethan
"""

#see on notebooks
import cv2
from dataset_albumentation import transform_train
from dataset_albumentation import transform_nolastic

image=cv2.imread('path_img')
mask=cv2.imread('path_mask', 0)

path_img='new_file'
path_mask='new_file'
cv2.imwrite(path_img+str(0)+'_augment.tif',image)
cv2.imwrite(path_mask+str(0)+'_augment.tif',mask) 
for i in range(1,5):
  augmented = transform_train(image=image, mask=mask)
  image_heavy = augmented['image']
  mask_heavy = augmented['mask']
  cv2.imwrite(path_img+str(i)+'_augment.tif',image_heavy)
  cv2.imwrite(path_mask+str(i)+'_augment.tif',mask_heavy)  
for i in range(5,10):
  augmented = transform_nolastic(image=image, mask=mask)
  image_heavy = augmented['image']
  mask_heavy = augmented['mask']
  cv2.imwrite(path_img+str(i)+'_augment.tif',image_heavy)
  cv2.imwrite(path_mask+str(i)+'_augment.tif',mask_heavy)