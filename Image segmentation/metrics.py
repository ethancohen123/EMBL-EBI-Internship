# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:16:21 2020

@author: ethan
"""


import numpy as np
import torch
import matplotlib.pyplot as plt


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    AC=active_contour_loss(target,pred)

    loss_bce_dice = bce * bce_weight + dice * (1 - bce_weight)
    
    loss=0.75*loss_bce_dice + 0.25*AC#put whatever weights you want

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['AC'] += AC*target.size(0)
    metrics['loss_bce_dice'] += loss_bce_dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def active_contour_loss(y_true, y_pred):
  '''
  y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
  weight: scalar, length term weight.
  '''
  # length term
  delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
  delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
  
  delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
  delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
  delta_pred = torch.abs(delta_r + delta_c) 

  epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
  lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
  
  # region term
  C_in  = torch.ones_like(y_pred)
  C_out = torch.zeros_like(y_pred)

  region_in  = torch.mean( y_pred     * (y_true - C_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
  region_out = torch.mean( (1-y_pred) * (y_true - C_out)**2 ) 
  region = region_in + region_out
  
  loss =  0.2*lenth + 0.8*region

  return loss




def compute_metrics(metrics, epoch_samples):
    computed_metrics = {}
    for k in metrics.keys():
        computed_metrics[k] = metrics[k] / epoch_samples
    return computed_metrics


def print_metrics(computed_metrics, phase):
    outputs = []
    for k in computed_metrics.keys():
        outputs.append("{}:{:4f}".format(k, computed_metrics[k]))

    print("\t{}-> {}".format(phase.ljust(5), "|".join(outputs)))


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()



def normalise_mask_set(mask, threshold):
  
  mask[mask > threshold] = 1
  mask[mask <= threshold] = 0
  return mask 


def normalise_mask(mask, threshold=0.5):
  
  mask[mask > threshold] = 1
  mask[mask <= threshold] = 0
  return mask    


def metrics_line(data):
    phases = list(data.keys())
    metrics = list(data[phases[0]][0].keys())

    i = 0
    fig, axs = plt.subplots(1, len(metrics))
    fig.set_figheight(6)
    fig.set_figwidth(6 * len(metrics))
    for metric in metrics:
        for phase in phases:
            axs[i].plot([i[metric] for i in data[phase]], label=phase)
        axs[i].set_title(metric)
        i += 1

    plt.legend()
    plt.show()