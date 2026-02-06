#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:25:04 2024

@author: munier
"""

import torch.optim as optim
import torch.nn as nn
import torch
import os

import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.parameters import get_GSPnP_params
import matplotlib.pyplot as plt
from deepinv.utils.plotting import plot
from torchvision.transforms.functional import rotate

from .convolution_Fresnel_2D import A_op, PSFGenerator2Dzernike_t



class Deblurring(nn.Module):
    def __init__(self, device=torch.device("cuda"), dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.psf_generator = None
        self.dpir_model = None
        self.y = None
        
    def set_variables(self, n_channels = 3, img_size = 200, psf_size = 31,
                      noise_level_img = 0.03, num_workers = 8, batch_size = 1,
                      amplitude_zernike = 0.05, i_min_zernike = 4, i_max_zernike = 12,
                      crop_size = None, max_iter_DPIR = 40, optim_method = "HQS"):
        
        self.n_channels = n_channels
        self.img_size = img_size
        self.psf_size = psf_size
        self.noise_level_img = noise_level_img
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.amplitude_zernike = amplitude_zernike
        self.i_min_zernike = i_min_zernike
        self.i_max_zernike = i_max_zernike
        self.max_iter_DPIR = max_iter_DPIR
        self.psf_generator = self.simple_psf_gen(self.amplitude_zernike, 
                                                 self.i_min_zernike, 
                                                 self.i_max_zernike, 
                                                 self.psf_size)
        
        if crop_size == None:
            crop_size = psf_size//2
        self.crop_size = crop_size
        self.y = None
        
        self._init_dpir_model(force = True, optim_method = optim_method)
        
        
    def psf_from_param(self, param):
        return self.psf_generator.generate_psf(param)
    
    def phys_from_param(self, param):
        psf = self.psf_from_param(param)
        return self.dinv_blur(psf)
    
    def Phi(self, theta):
        psf = self.psf_from_param(theta)
        phys = self.dinv_blur(psf)
        x1 = self.dpir_model(self.y, phys)
        y1 = A_op(x1, psf) 
        #y1 = phys(x1)
        y1_crop = y1[:, :, self.crop_size:self.img_size-self.crop_size, 
                  self.crop_size:self.img_size-self.crop_size]
        
        del x1, y1
        return y1_crop
    
    def set_model(self, y):
        self.y = y
    
    def generate_random_coeffs(self, x = None, param = None):
        if param == None:
            param = self.psf_generator.generate_random_coeffs()
        y0 = self.convolve(x, param)
        y = y0 + torch.randn_like(y0) * self.noise_level_img
        
        return x, param, y
    
    def simple_psf_gen(self, amplitude_zernike, i_min_zernike, 
                       i_max_zernike, psf_size):
        list_param = []
        for i_param in range(i_min_zernike, i_max_zernike):
            list_param.append("Z"+str(i_param))
        
        n_param = len(list_param)
        min_coeff_zernike = [-amplitude_zernike] * n_param
        max_coeff_zernike = [amplitude_zernike] * n_param
        
        psf_generator = PSFGenerator2Dzernike_t(list_param=list_param, psf_size=psf_size,
                                                     min_coeff_zernike=min_coeff_zernike, 
                                                     max_coeff_zernike=max_coeff_zernike,
                                                     device=self.device, dtype=self.dtype)
        return psf_generator
    
    #Convolution
    def convolve(self, x, theta):
        p = self.phys_from_param(theta)
        return p(x)
    
    #Convolution transposed
    def convolve_T(self, y, theta):
        p = self.phys_from_param(theta)
        return p.A_adjoint(y)
    
    #Deconvolution
    def deconvolve(self, y, theta):
        p = self.phys_from_param(theta)
        return self.dpir_model(y, p)
    
    def dinv_blur(self, psf):
        return dinv.physics.BlurFFT(
            img_size=(self.n_channels, self.img_size, self.img_size),
            filter=psf[None,None,:,:],
            device=self.device,
            noise_model=dinv.physics.GaussianNoise(sigma=0),
        )
    
    def _init_dpir_model(self, force = False, optim_method = "HQS"):
        # Set up the DPIR algorithm to solve the inverse problem.
        # --------------------------------------------------------------------------------
        # This method is based on half-quadratic splitting (HQS).
        # The algorithm alternates between a denoising step and a data fidelity step, where
        # the denoising step is performed by a pretrained denoiser :class:`deepinv.models.DRUNet`.
        
        if self.dpir_model == None or force:
            # load specific parameters for DPIR
            lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params("deblur", self.noise_level_img)
            params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
            early_stop = False  # Do not stop algorithm with convergence criteria
            
            if self.max_iter_DPIR != None:
                max_iter = self.max_iter_DPIR
            
            # Select the data fidelity term
            data_fidelity = L2()
            
            # Specify the denoising prior
            denoiser=DRUNet(pretrained=None, device=self.device).to(self.dtype)
            denoiser.train()

            ckpt_path = "../../model_zoo/drunet_color.pth"
            
            temp_pth = torch.load(str(ckpt_path), map_location=self.device)
            denoiser.load_state_dict(temp_pth)
            prior = PnP(denoiser)
            
            for p in prior.parameters():
                p.requires_grad_(False)
            
            # instantiate the algorithm class to solve the IP problem.
            self.dpir_model = optim_builder(
                iteration = optim_method,
                prior=prior,
                data_fidelity=data_fidelity,
                early_stop=early_stop,
                max_iter=max_iter, #max_iter,
                verbose=True,
                params_algo=params_algo,
            ).to(self.dtype)


#%%
if __name__ == "__main__":
    
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    img_size = 200
    batch_size = 1
    num_workers = 8
    
    # THE DEBLURRING MODEL
    deblur_model = Deblurring()
    deblur_model.set_variables(img_size = img_size, batch_size = batch_size,
                               num_workers = num_workers)
    x, param, y = deblur_model.generate_random_coeffs()
    deblur_model.set_model(y)
    Phi = deblur_model.Phi
    
    param.requires_grad = True
    #objective = torch.sum(deblur_model.deconvolve(y, param)**2)
    objective = torch.sum(Phi(param)**2)
    objective.backward(retain_graph=False)
    
    grd = param.grad
    
    param.requires_grad = False
    
    with torch.no_grad():
        estim = deblur_model.deconvolve(y, param)
        plot(estim)
        
        y2 = Phi(param)
        plot(y2)
        plot(y)
        
    f = lambda theta : torch.sum(Phi(theta)**2)
    i_optim = 0
    bfgs_cv = False

    history_lbfgs = []

    # L-BFGS STEPS
    def closure():
        lbfgs.zero_grad()
        objective = f(param_test)
        objective.backward(retain_graph=False)
        return objective

    param_test = param.ravel().clone().detach()
    param_test.requires_grad = True

    lbfgs = optim.LBFGS([param_test],
                        lr=1.,
                        tolerance_grad=1e-5,
                        tolerance_change=1e-5,
                        history_size=10,
                        max_iter=10,
                        line_search_fn="strong_wolfe")

    
    prev_loss = float('inf')
    history_lbfgs = []
    
    for i_optim in range(10):
        print(i_optim)
        loss = lbfgs.step(closure)
        with torch.no_grad():
            history_lbfgs.append(loss.item())
    
    
    