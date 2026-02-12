#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:20:11 2024

@author: munier
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder

from tqdm import tqdm

from jackpot.direct_model import Model
from jackpot.GSPnP_jvp import GSDRUNet
from jackpot.utils import tensor_empty_cache

import numpy as np

import os
from pathlib import Path

from deepinv.optim.prior import RED
import imageio.v3 as iio
from PIL import Image
import torchvision.transforms as transforms

def singsolver_get_filename(direct_model, singular_solver):
    """
    Get the file name where all saves from singular solver are stored

    Parameters
    ----------
    direct_model : DirectModel
    singular_solver : SingularSolver

    Returns
    -------
    str
        filename

    """
    return (direct_model.save_rootname
                / Path(direct_model.expe_title
                       + "_" + singular_solver
                       + "_sing_vals.pth"))

def manifold_solver_get_filename(direct_model, i_sing):
    """
    Get the file name where all saves from manifold solver are stored

    Parameters
    ----------
    direct_model : DirectModel
    i_sing : int
        i th singular value. (from 0 to N-1)

    Returns
    -------
    str
        filename

    """
    return direct_model.save_rootname / \
        Path(direct_model.expe_title + f"_dir_{i_sing+1}.pth")

def manifold_solver_2_get_filename(direct_model, dim):
    """
    Get the file name where all saves from manifold solver are stored 
        in the case of a grid of dimension dim

    Parameters
    ----------
    direct_model : DirectModel
    dim : int
        dimension of the approximation manifold

    Returns
    -------
    str
        filename

    """
    return direct_model.save_rootname / \
        Path(direct_model.expe_title + f"_dim_{dim}.pth")
        
def load_model(model, image_type, img_size, n_channels, recompute_map, plot_img):
    """
    Load the direct model (direct map, jacobian functions, initial point x0, map estimator)

    Parameters
    ----------
    model : str
        model name.
    image_type : str
        image name.
    img_size : int
        image size.
    n_channels : int (1 or 3)
        number of channels.
    recompute_map : boolean
        If True recompute the map estimator, if not recompute it only if not already stored.
    plot_img : boolean
        Plot x0, x_map, x_backward = AT y, and the intermediary estimations of the map.

    Returns
    -------
    direct_model : DirectModel
    jvp_fn : function
        jacobian vector product function.
    vjp_fn : function
        vector jacobian product function.
    x0 : tensor
        initial point.
    x_map : tensor
        map estimator.

    """
    direct_model = DirectModel()
    
    direct_model.choose_model(model, image_type,
                              img_size=img_size, n_channels=n_channels,
                              recompute_map=recompute_map)

    jvp_fn, vjp_fn, x0, x_map, _ = direct_model.get_all(plot_img)

    direct_model.empty_cache()
    
    return direct_model, jvp_fn, vjp_fn, x0, x_map

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

# IMPORT GRAD LOG PRIOR AND LIKELIHOOD
class GradStepPnP(RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)

class DirectModel():
    """
    Class of direct model in order to define standard convolution and irm models.
    Main available functions are:
        - define the model (choose_model)
        - get main functions (get_all)
        - erase model (empty_cache, delete_model)
        
    Other functions are packed inside those functions and are not usually used.
    """
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.AT = None

    def choose_model(self, model="motion_convol", image_type="fastmri",
                     img_size=256, n_channels=3, recompute_map=False,
                     plot_img=False, expe_title = None):
        """
        Define the model to generate.

        Parameters
        ----------
        model : str, optional
            model name. The default is "motion_convol".
        image_type : str, optional
            image name. The default is "fastmri".
        img_size : int, optional
            image size. The default is 256.
        n_channels : int, optional
            number of channels. The default is 3.
        recompute_map : bool, optional
            If True recompute the map estimator, 
            if not recompute it only if not already stored.
            The default is False.
        plot_img : bool, optional
            Plot x0, x_map, x_backward = AT y, and the intermediary estimations of the map.
            The default is False.

        Returns
        -------
        None. Set the parameters inside the direct_model object.

        """

        assert model in ["motion_convol", "gaussian_convol", "partial_fft", "small_kernel_convol"]

        self.model = model
        self.image_type = image_type
        self.plot_img = plot_img

        ###############  ALL THE PARAMETERS  #######################################
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # Load and save parameters
        if expe_title == None:
            self.expe_title = self.model
            self.expe_title = self.expe_title + "_" + self.image_type
            self.expe_title = self.expe_title + f"_{img_size}"
        else:
            self.expe_title = expe_title
        self.save_rootname = Path("./saves/")
        self.plot_rootname = Path("./plots/")
        os.makedirs(self.save_rootname, exist_ok=True)
        os.makedirs(self.plot_rootname, exist_ok=True)

        # Model parameters
        self.img_size = img_size
        self.n_channels = n_channels
        self.noise_thres = 0.01

        # load specific parameters for GSPnP
        lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params("deblur", 
                                                            self.noise_thres)
        
        if model == "motion_convol":
            lamb = 0.15
        elif model == "gaussian_convol":
            lamb = 0.15
        elif model == "partial_fft":
            lamb = 0.1
        
        self.recompute_map = recompute_map
        self.sigma_denoiser = sigma_denoiser #0.1  
        self.l_l2 = 1. / lamb # 1e2
        self.input_shape = (1, self.n_channels, self.img_size, self.img_size)

        # Gradient step parameters
        self.init_grad_step = self.recompute_map
        self.n_iter = 50
        
        # if model == "partial_fft":
        #     self.stepsize = 0.000001 / self.l_l2  # 1e-1
        # else:
        #     self.stepsize = 2e-6 # 0.1 / self.l_l2 # 0.5 / self.l_l2  # 1e-1
        self.stepsize = stepsize #/ 5
        
        # Prior
        self.drunet = None
        self.prior = None

        #######################################################################

    def get_all(self, plot_img=None, method = "manual_gd"):
        """
        Provide mains functions and objects of the model 
        (jvp_fn, vjp_fn, x0, x_map, apply_P)
        
        Be aware that self.A (and self.AT) is the direct linear model operator (and its transposed)
        While self.Phi is the gradient of the loss function:  
                0.5 * l_l2 * norm(Ax - y)**2 + grad log_prior
        
        Parameters
        ----------
        plot_img : bool, optional
            Plot x0, x_map, x_backward = AT y, and the intermediary estimations of the map.
            The default is None.

        Returns
        -------
        jvp_fn : function
            jacobian vector product function.
        vjp_fn : function
            vector jacobian product function.
        x0 : tensor
            initial point.
        x_map : tensor
            map estimator.
        apply_P : None
            Preconditioner.

        """
        if plot_img != None:
            self.plot_img = plot_img

        ### IMPORT IMAGE ###
        self.x = self.load_expe_image()
        if self.plot_img:
            plot(self.x, titles=["Initial image"])
        
        ### IMPORT THE DIRECT MODEL ###
        self.import_direct_model()        

        ### IMPORT GRAD LOG PRIOR AND LIKELIHOOD ###
        self.init_grad_prior_fn()

        self.y_true = self.A(self.x)
        self.y = self.y_true + self.noise_thres * \
            torch.randn_like(self.y_true)

        self.x_back = self.AT(self.y).detach().clone()

        if self.plot_img:
            if self.model == "partial_fft":
                plot(torch.log(1 + torch.abs(torch.fft.fftshift(
                    torch.view_as_complex(self.y)
                ))),
                    titles=['y IRM'])
            else:
                plot(self.y, titles=['y output'])
        
        ### GRADIENT STEPS ###
        self.get_or_compute_map(method = method)

        if self.plot_img:
            plot(self.x_back, titles=["Back projection"])
            plot(self.x_map, titles=['Estimation MAP'])

        apply_P = None
        # ### PRECONDITIONNER ###
        # if self.model == "partial_fft":
        #     apply_P = None
        # else:
        #     apply_P = self.preconditioner_of_convol(self.kernel)

        ### GET JVP AND VJP FUNCTIONS ###
        x0 = self.x.clone()
        x_map = self.x_map.clone()

        self.Phi = Model(self._Phi, x0, parallel=False)

        self.Phi.set_jacobian_mult_fn(x_map)
        jvp_fn, vjp_fn = self.Phi._jvp_fn, self.Phi._vjp_fn

        return jvp_fn, vjp_fn, x0, x_map, apply_P

    def empty_cache(self):
        """Empty memory cache while letting the main function available"""
        self.x = None
        self.x_back = None
        self.x_map = None
        self.kernel = None

    def delete_model(self):
        """
        Delete the model and empty all memory used
        """
        if self.Phi != None:
            self.Phi.empty_jac()
        self.prior = None
        self.Phi = None
        self.drunet = None
        tensor_empty_cache(self.x, self.x_back, self.x_map, self.kernel, self.y)
        self.x = None
        self.x_back = None
        self.x_map = None 
        self.kernel = None 
        self.y = None
        self.ip_solver = None
        self.A = None
        self.AT = None

    def reshape_image(self, x):
        ### SOME CHECK ON THE SIZE OF THE IMAGE ###
        if type(x) != torch.Tensor:
            x = torch.tensor(x, **self.factory_kwargs)
        else:
            x = x.to(**self.factory_kwargs)
        
        if x.shape == (self.img_size, self.img_size, self.n_channels):
            x = x[None, ...]
            x = x.permute((0, 3, 1, 2))
        elif x.shape == (self.img_size, self.img_size):
            x = x[None, None, ...]
            x = x * torch.ones((1, self.n_channels, 1, 1),
                               **self.factory_kwargs)
        elif x.shape == (self.n_channels, self.img_size, self.img_size):
            x = x[None, ...]
        assert self.input_shape == x.shape
        return x

    def import_image(self, filename):
        x = iio.imread(filename)
        img = Image.fromarray(x).resize((self.img_size, self.img_size))
        img_tensor = (transforms.PILToTensor())(img)
        return self.reshape_image(img_tensor)

    def load_expe_image(self):
        """
        Load the saved images
        """
        load_image_dict = {}
        load_image_dict["fastmri"] = Path("im_99_denoised.png")
        load_image_dict["fastmri2"] = Path("im_55.png")
        load_image_dict["fastmri3"] = Path("im_77.png")
        load_image_dict["fastmri4"] = Path("im_178.png")
        load_image_dict["fastmri5"] = Path("im_229.png")
        load_image_dict["butterfly"] = Path("butterfly.png")
        load_image_dict["leaves"] = Path("leaves.png")
        load_image_dict["starfish"] = Path("starfish.png")
        load_image_dict["girafes"] = Path("girafes.png")
        load_image_dict["elephant"] = Path("elephant.png")
        load_image_dict["bell"] = Path("bell.png")
        load_image_dict["bell_zoom"] = Path("bell_zoom.png")
        load_image_dict["sheeps"] = Path("sheeps.png")
        load_image_dict["teddy_bears"] = Path("teddy_bears.png")
        load_image_dict["barbara"] = Path("barbara.jpg")
        load_image_dict["drepanocytose"] = Path("cells_drepanocytose.jpg")
        load_image_dict["blood"] = Path("blood_cells.png")
        
        if self.image_type in load_image_dict.keys():
            name = load_image_dict[self.image_type]
            filename = Path("./dataset/") / name
            if not(filename.is_file()):
                print(filename, "is not a file")
            x = self.import_image(filename=filename)
        else:
            print(f"------ {self.image_type} is not an image type!")
        return x

    def get_or_compute_map(self, method = "manual_gd"):
        """
        Compute the map estimator of the loss function 
            0.5 * l_l2 * norm(Ax - y)**2 + log_prior(x)
        

        Parameters
        ----------
        method : str, optional
            Optimization method. Either:
                - "sgd": stochastic gradient descent from pytorch
                - "pgd": Projected gradient descent from deepinv
                - "manual_gd": Gradient descent with Armijo line search 
                - "lbfgs": L-BFGS from pytorch
            The default is "manual_gd".

        Returns
        -------
        x : tensor
            initial input x.
        x_map : tensor
            MAP estimator.
        """
        filename_init = self.save_rootname / \
            Path(self.expe_title + "_x_init.pth")

        is_init_image_exists = filename_init.exists()

        if not (is_init_image_exists):
            print(filename_init, "is not a file")

        if self.init_grad_step or not (is_init_image_exists):
            
            if method == "sgd":
                self.x_map = self.x_map_sgd(self.x_back, self.y)
            elif method == "pgd":
                with torch.no_grad():
                    self.x_map = self.ip_solver(self.y, self.A)
            elif method == "manual_gd":
                self.x_map = self.x_map_manual_gd(0.5 * torch.ones_like(self.x_back), self.y) # self.x_back
            elif method == "lbfgs":
                self.x_map = self.x_map_lbfgs(self.x_back, self.y)
            
            torch.save((self.x, self.x_map, self.y), filename_init)
        else:
            self.x, self.x_map, self.y = torch.load(filename_init)
            # self.x.to(**self.factory_kwargs)
            # self.x_map.to(**self.factory_kwargs)
            # self.y.to(**self.factory_kwargs)

        filename = self.save_rootname / Path('temp_x_map.png')
        if self.n_channels == 3:
            iio.imwrite(filename, rescaleUINT8(
                self.x_map.view(self.input_shape)[0, ...].permute(1, 2, 0)))
        filename = self.save_rootname / Path('temp_x.png')
        if self.n_channels == 3:
            iio.imwrite(filename, rescaleUINT8(
                self.x.view(self.input_shape)[0, ...].permute(1, 2, 0)))        

        return self.x, self.x_map

    def x_map_lbfgs(self, x0, y):
        lr = 1e-1
        tol = 1e-5
        history_size = 20
        max_iter_per_step = 100
        line_search = "strong_wolfe"

        x = x0.clone().detach()
        x.requires_grad = True

        lbfgs = optim.LBFGS([x],
                            lr=lr,
                            tolerance_grad=tol,
                            tolerance_change=tol,
                            history_size=history_size,
                            max_iter=max_iter_per_step,
                            line_search_fn=line_search)

        t = tqdm(range(self.n_iter))
        # L-BFGS STEPS

        def closure():
            lbfgs.zero_grad()

            with torch.no_grad():
                grad = self.grad_prior_fn(x, self.sigma_denoiser)
                x.grad = grad

                loss = self.data_loss(x, y)

                t.set_description(
                    f"lbfgs map : {grad.norm().item():.5f}, loss: {loss.item():.3f}"
                )

                return loss

        for k in t:
            loss = lbfgs.step(closure)

            if (k+1) % 500 == 0:
                x_n = x / x.max()
                if self.plot_img:
                    plot(x_n, titles=["Intermediary image"])

        # Empty lbfgs memory cache !
        lbfgs.zero_grad()
        
    def x_map_manual_gd(self, x0, y, verbose = True):
        # Armijo conditions
        tau = 0.5
        c = 0.9
        lr = self.stepsize /10
        x = x0.clone().detach()
        if verbose:
            t = tqdm(range(self.n_iter))
        else:
            t = range(self.n_iter)
        for k in t:
            lr = lr * 1.5
            x.requires_grad = True
            x.grad = None
            data_ls = self.data_loss(x, y)
            data_ls.backward()
            
            with torch.no_grad():
                prior_loss = self.prior_fn(x, self.sigma_denoiser)
                x.grad.data += self.grad_prior_fn(x, self.sigma_denoiser)
                m = torch.sum(x.grad**2)
            
                x_temp = x - lr * x.grad 
                loss_temp = self.data_loss(x_temp, y) + self.prior_fn(x_temp, 
                                                                 self.sigma_denoiser)
                
                while loss_temp > data_ls + prior_loss - c * lr * m:
                    lr = tau * lr
                    x_temp = x - lr * x.grad 
                    loss_temp = self.data_loss(x_temp, y) + self.prior_fn(x_temp, 
                                                                      self.sigma_denoiser)
                
                x = x_temp.detach()
                
                if verbose:
                    t.set_description(
                        f"lr: {lr:.3e}, loss: {loss_temp.item():.3f}, grad norm2: {m:.3e}, data_loss: {data_ls.item():.3e}, prior_loss: {prior_loss.item():.3e}")
    
                if self.plot_img:
                    if (k+1) % 50 == 0:
                        plot(x, titles=["Intermediary image"])
        return x

    def x_map_sgd(self, x0, y):
        x = x0.clone().detach()

        x.requires_grad = True
        
        sgd = optim.SGD([x], lr=self.stepsize, momentum=0.9)
        t = tqdm(range(self.n_iter))
        for k in t:
            sgd.zero_grad()
            loss = self.data_loss(x, y)
            loss.backward()
            
            with torch.no_grad():
                x.grad.data += self.grad_prior_fn(x, self.sigma_denoiser)
            
            sgd.step()

            t.set_description(
                f"loss: {loss.item():.3f}, grad norm: {x.grad.norm():.3e}")

            if self.plot_img:
                if (k+1) % 25 == 0:
                    x_n = x / x.max()
                    plot(x_n, titles=["Intermediary image"])
        sgd.zero_grad()
        return x

    def _Phi(self, x):
        """
        Compute the gradient of the loss function:
            Phi(x) = 0.5 * l_l2 * norm(Ax - y)**2 + log_prior(x)
        y is taken from the stored self.y
        
        Parameters
        ----------
        x : tensor

        Returns
        -------
        Phi(x): tensor
        """
        return (self.l_l2 * self.AT(self.A(x) - self.y)
                + self.grad_prior_fn(x, self.sigma_denoiser))

    def data_loss(self, x, y):
        """
        Data part of the loss function:
            0.5 * l_l2 * norm(Ax - y)**2

        Parameters
        ----------
        x : tensor
            input.
        y : tensor
            output.

        Returns
        -------
        data_ls : float
            Data loss.

        """
        data_ls = self.l_l2 / 2 * torch.sum(torch.abs(self.A(x) - y)**2)
        return data_ls

    def prior_fn(self, x, sigma):
        if self.n_channels == 3:
            return self.prior.g(x.view(self.input_shape), sigma)
        elif self.n_channels == 1:
            return self.prior.g(x.view(self.input_shape) * 
                                torch.ones((1,3,1,1), **self.factory_kwargs), sigma)

    def grad_prior_fn(self, x, sigma):
        if self.n_channels == 3:
            return self.prior.grad(x.view(self.input_shape), sigma)
        elif self.n_channels == 1:
            return torch.sum(self.prior.grad(x.view(self.input_shape) * 
                            torch.ones((1,3,1,1), **self.factory_kwargs), sigma), 
                             axis = 1, keepdim = True)

    def init_grad_prior_fn(self):
        """
        Load the prior and grad prior from Hurault.
        It is precisely a Gradient-Step Denoiser prior and the denoiser is a DRUNet.
        Either download it or take it from store.
        """
        
        self.drunet = GSDRUNet(pretrained="download").to(**self.factory_kwargs)
        self.prior = GradStepPnP(denoiser=self.drunet)
        
        # load specific parameters for GSPnP
        lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params("deblur", 
                                                            self.noise_thres)
        stepsize = stepsize / 5
        
        params_algo = {
            "stepsize": stepsize,
            "g_param": sigma_denoiser,
            "lambda": lamb,
        }
        early_stop = False  # Do not stop algorithm with convergence criteria
        
        # Select the data fidelity term
        data_fidelity = L2()
        
        # instantiate the algorithm class to solve the IP problem.
        self.ip_solver = optim_builder(
            iteration="PGD",
            prior=self.prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=True,
            params_algo=params_algo,
        )

    def set_A_fn(self):
        """
        Linear model (either convolution or mri) defined from the model.

        Returns
        -------
        A: function
            linear model.

        """
        if self.model == "partial_fft":
            self.A = self._A_fn_fft
        else:
            convolve = dinv.physics.BlurFFT(
                img_size=(self.n_channels, self.img_size, self.img_size),
                filter = self.kernel, device=self.device
            )
            self.A = convolve
        
        
    def _A_fn_fft(self, x):
        return torch.view_as_real(torch.fft.fft2(x) * self.mask)

    def set_AT_fn(self):
        """
        Transposed of the linear model.

        Returns
        -------
        AT: function
            transposed linear model.

        """
        if (self.model == "motion_convol" or self.model == "gaussian_convol" 
            or self.model == "small_kernel_convol"):
            self.AT = self.A.A_adjoint
        elif self.model == "partial_fft":
            self.AT = self._AT_fn_fft

    def _AT_fn_fft(self, y):
        return torch.real(torch.fft.ifft2(self.mask * torch.view_as_complex(y)))
        
    def import_direct_model(self):
        """
        Import the kernel or the mask of the model to define the function self.A and self.AT

        Returns
        -------
        A: function
            linear model.
        AT: function
            transposed linear model.

        """
        if self.model == "motion_convol":
            # Motion blur
            psf_size = 11
            self.kernel = torch.zeros(
                (1, 1, psf_size, psf_size), **self.factory_kwargs)
            self.kernel[0, 0, psf_size//2, :] = 1./psf_size

        elif self.model == "gaussian_convol":
            # Gaussian Convolution
            self.kernel = dinv.physics.blur.gaussian_blur(
                sigma=(2, 2), angle=0.0)

        elif self.model == "partial_fft":
            self.mask = torch.zeros((1, 1, self.img_size, self.img_size),
                                    **self.factory_kwargs)
            n0, n1 = self.img_size, self.img_size
            nrev = 20

            ntraj = 8
            if self.img_size == 256:
                ntraj = 4
            if self.img_size == 320:
                ntraj = 8
            if self.img_size == 512:
                ntraj = 8
            if self.img_size == 128:
                ntraj = 3

            pi = torch.pi
            T = torch.linspace(0, 1, 32 * n0, **self.factory_kwargs)

            for k in range(ntraj):
                traj0 = ((T**2 *
                          torch.cos(2 * pi * T * nrev + 2 * k * pi / ntraj)
                          * n0/2 * 2**0.5))
                traj1 = ((T**2 *
                          torch.sin(2 * pi * T * nrev + 2 * k * pi / ntraj)
                          * n1/2 * 2**0.5))
                for ii, jj in zip(traj0 + n0//2, traj1 + n1//2):
                    if ii >= 0 and jj >= 0 and ii < n0 and jj < n1:
                        self.mask[0, 0, int(ii), int(jj)] = 1

            del T, traj0, traj1

            if self.plot_img:
                plot(self.mask, titles=["Mask"])
            print(torch.sum(self.mask) / (n0*n1) * 100)
            self.mask = torch.fft.ifftshift(self.mask)
        
        elif self.model == "small_kernel_convol":
            # Psf definition such that its Fourier transform has only 2 small values
            psf = torch.zeros((self.img_size, self.img_size), **self.factory_kwargs)
            psf[0,0] = 1
            psf[1,0] = 1
            psf[1,1] = 1/10
            psf[0,1] = 1
            psf[0,2] = 1
            psf[0,3] = 1
            psf /= psf.sum()
            psf = torch.fft.fftshift(psf[None,None])
            fpsf = torch.fft.fft2(psf)
            afpsf = torch.abs(fpsf)
            thr = afpsf.min() * 1.001
            fpsf[afpsf > thr] = fpsf[afpsf > thr] + thr * 100 * (
                fpsf[afpsf > thr] / afpsf[afpsf > thr])
            psf = torch.real(torch.fft.ifft2(fpsf))
            psf /= psf.sum()
            fpsf = torch.fft.fft2(psf)
            
            self.kernel = psf
            
        self.set_A_fn()
        self.set_AT_fn()

    def preconditioner_of_convol(self, kernel):
        """
        Preconditionning

        Parameters
        ----------
        kernel : tensor
            Kernel against which reduce the intensity of the inner norm.

        Returns
        -------
        apply_P: function
            Precondition function.

        """
        
        # Generate kernel
        ker_size = kernel.shape[2]
        pad_size = (self.img_size - ker_size)//2

        if pad_size != 0:
            kernel = F.pad(kernel, (pad_size+1, pad_size, pad_size+1, pad_size),
                           mode='constant', value=0)
        kernel = kernel.to(**self.factory_kwargs)
        shift_kernel = torch.fft.ifftshift(kernel)

        # Preconditioned matrix
        self.P = 1 / (torch.abs(torch.fft.fft2(shift_kernel))
                      ** 2 + self.eps_tykh)

        tensor_empty_cache(kernel, shift_kernel)
        del kernel, shift_kernel

        def apply_P(u):
            fft_u = torch.fft.fft2(u.view(self.input_shape))
            return torch.real(torch.fft.ifft2(fft_u * self.P)).ravel()

        return apply_P

    def plot_singular_images(self, X, sing_vals=None):
        """
        Plot the singular values

        Parameters
        ----------
        X : tensor of shape img_shape + (K,)
            singular vectors.
        sing_vals : tensor of shape (K,), optional
            singular values. The default is None.

        Returns
        -------
        None.

        """
        k_vect = X.shape[-1]

        vec_shape = self.input_shape
        vec_shape = (k_vect,) + vec_shape[1:]

        for k in range(k_vect):
            img_k = X.T.reshape(vec_shape)[k, ...][None, ...]
            if sing_vals is None:
                plot(img_k, titles=[f"Singular vector {k}"])
            else:
                plot(img_k, titles=[
                     f"Singular vector {k} -- {sing_vals[k]:1.2e}"])
            modk = torch.abs(torch.fft.fft2(img_k))
            plot(modk, titles=[f"FFT singular vector {k}"])

        tensor_empty_cache(img_k, modk)
        del img_k, modk
