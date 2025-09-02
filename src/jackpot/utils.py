import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


import os


def send_to_cpu(x):
    return x.clone().detach().to("cpu")

def tensor_empty_cache(*X):
    for x in X:
        if isinstance(x, torch.Tensor):
            # Delete also graph and gradient of the tensor
            if x.grad_fn != None:
                x.detach_()
                
                
class FlatForward(nn.Module):
    def __init__(self, operator, input_shape):
        super().__init__()
        self.operator = operator
        self.input_shape = input_shape

    def forward(self, x):
        return self.operator(x.view(self.input_shape)).ravel()
    
    

def default_plot_styles():
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)),
                  (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return linestyles, colors


def save_this_plot(image_directory, image_name, axis=None, title="", filetype=".png", clear=True):
    image_path = os.path.join(image_directory, image_name + filetype)
    # Clear the current figure for the next iteration
    if axis != None:
        plt.axis(axis)
    if title != "":
        plt.title(title)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    # Clear the current figure for the next iteration
    if clear:
        plt.clf()
    else:
        plt.show()


def save_image(x, image_directory, image_name, axis='off', title="", filetype=".png", clear=True):
    if len(x.shape) == 2:
        x = x[None, ...]
    assert len(x.shape) == 3
    assert x.shape[0] in [1, 3]

    image_path = os.path.join(image_directory, image_name + filetype)

    plt.imshow(x.permute(1, 2, 0).tolist())
    plt.axis(axis)
    plt.title(title)
    # Save the image as PNG
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    # Clear the current figure for the next iteration
    if clear:
        plt.clf()
    else:
        plt.show()

def rescaleUINT8(x):
    """
    set an image to UINT8 format

    Parameters
    ----------
    x : tensor
        image.

    Returns
    -------
    x_out : tensor
        image of UINT8 format.

    """
    x = x - x.min()
    x = x / x.max()
    x = (x * 255).detach().cpu().numpy()
    return np.uint8(x)

def clipUINT8(x):
    """
    set an image to UINT8 format by first clipping in [0,1]

    Parameters
    ----------
    x : tensor
        image.

    Returns
    -------
    x_out : tensor
        image of UINT8 format.

    """
    x = torch.clip(x, min=0, max = 1)
    x = (x * 255).detach().cpu().numpy()
    return np.uint8(x)