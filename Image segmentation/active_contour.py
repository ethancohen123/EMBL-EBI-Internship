# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:10:18 2020

@author: ethan
"""


from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


#ENTER THE IMAGE IN GRAYSCALE ,named imgray:
    
def GAC(imgray):
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()
    
    
    
    # Morphological GAC
    image = img_as_float(imgray)
    gimage = inverse_gaussian_gradient(image)
    
    # Initial level set
    
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 150, init_level_set='circle',
                                               smoothing=1, balloon=-1,
                                               threshold='auto',
                                               iter_callback=callback)
    
    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological GAC segmentation", fontsize=12)
    
    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[1].contour(evolution[50], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 50")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 150")
    ax[1].legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax[1].set_title(title, fontsize=12)
    
    fig.tight_layout()
    plt.show()


def ACWE(imgray):
    image = img_as_float(imgray)

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 25, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()
    
    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
    
    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[15], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 15")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 20")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)
    
    
    
    
    
    fig.tight_layout()
    plt.show()