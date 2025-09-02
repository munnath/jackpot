#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:33:04 2024

@author: munier
"""

import torch
import torch.nn as nn

from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt


from utils import tensor_empty_cache, FlatForward
from torch_lobpcg import lobpcg

"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRMAWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""


def generate_rand_ortho_vects(N, n_vects, **factory_kwargs):
    """
    Generate random n_vects orthogonal vectors of size N

    Parameters
    ----------
    N : int
        length of the desired orthogonal vectors
    n_vects : int
        number of desired vectors.
    **factory_kwargs : dictionnary
        which has the keys "device" and "dtype"

    Returns
    -------
    X_ortho : tensor of shape (N, n_vects)
        orthogonal vectors.

    """
    assert N >= n_vects
    X = torch.randn((N, n_vects,), **factory_kwargs)
    X = X / torch.sum(X**2, axis=0)**0.5
    X_ortho, _, _ = torch.linalg.svd(X, full_matrices=False)

    del X

    return X_ortho


def sing_solver_load_experiment(filename, device="cuda", dtype=torch.float32):
    factory_kwargs = {"device": device, "dtype": dtype}
    (X, sing_vals, hist_ray, hist_sing, hist_time) = torch.load(filename)
    X = X.to(**factory_kwargs)
    sing_vals = sing_vals.to(**factory_kwargs)
    return (X, sing_vals, hist_ray, hist_sing, hist_time)


def sing_solver_plot_history(hist_ray, hist_sing, hist_time):
    plt.figure()

    plt.semilogy(hist_time, hist_ray, "-+")
    plt.semilogy(hist_time, hist_sing, "-+")

    plt.xlabel("time in s")

    plt.legend(["Rayleight quotient", "singular values"])

    plt.show()


class MultiApply(nn.Module):
    def __init__(self, fn, parallel=False):
        """
        Evaluate multiple times the function fn

        Parameters
        ----------
        fn : function
        parallel : bool
            using vmap to parallelize. The default is False.
        """
        super().__init__()
        self.fn = fn
        self.parallel = parallel

        if self.parallel:
            self.vmap_fn = torch.vmap(
                self.fn, in_dims=-1, out_dims=-1, randomness="same")

    def forward(self, X):
        """
        Parameters
        ----------
        X : tensor of shape (x.shape, n_instance) 
            where x is an example of input for fn, and n_instance is int.
        Returns
        -------
        Y
            Evaluations of the fn(X)
        """
        if self.parallel:
            return self.vmap_fn(X)
        else:
            assert X.ndim >= 2
            # X shoud be of shape (N, k_vect)
            fn_vals = []

            for i in range(X.shape[-1]):
                fn_val = self.fn(X[..., i])
                fn_vals.append(fn_val[..., None])

            del fn_val

            return torch.cat(fn_vals, dim=-1)

    def erase(self):
        if self.parallel:
            del self.vmap_fn


def singular_vectors(A, method = "svd", vector_indexes = (-1,)):
    """
    A: matrix of shape M x N
    threshold: float
    with_singular_values: boolean
    method: string
    ##################################
    If with_singular_values == True:
        Kernel
    Else:
        (Kernel, Singular_values)
    
    method:
        - "svd": compute the SVD of the Jacobian and deduce the kernel
        - "svd_ATA": compute the SVD of the matrix J^T J and deduce the kernel
    where:
     - Kernel is a matrix of shape D x N with D>=0 the dimension of the kernel 
       It corresponds to the singular vectors with singular values < threshold using the whole SVD computation
     - Singular_values is a tensor of shape M with the singular values 
    """
    assert (type(vector_indexes) == tuple)
    assert len(vector_indexes) > 0
    with torch.no_grad():
        (M,N) = A.shape
        if method == "svd":
            _, full_S_, full_V_ = torch.linalg.svd(A, full_matrices = True)
            
        if method == "svd_ATA":
            _, full_S_, full_V_ = torch.linalg.svd(A.T @ A, full_matrices = True)
            full_S_ = torch.sqrt(full_S_)
        
        if M < N:
            full_S_ = torch.cat([full_S_,torch.zeros((N - M,), device = A.device, dtype = A.dtype)])[:N,...]
        
        
        full_kernel = (full_V_[vector_indexes,:])
        
        
        #Empty the memory 
        full_V_ = None
        torch.cuda.empty_cache()
        #Renormalize the elements of kernel
        full_kernel_norm = full_kernel / torch.linalg.norm(full_kernel, axis=(1,))[:,None]
        
        return (full_kernel_norm, full_S_)




class SingularSolver(nn.Module):
    def __init__(self, Mat, MatT, x_example, Matx_example=None,
                 k_sing_vals=1, X_init=None, parallel=False):
        """
        WARNING: BY DEFAULT EVERY FUNCTIONS WILL BE VIEWED AS MATRIX 
            MEANING THAT INPUTS X AND OUTPUT AX WILL BE VIEWED AS VECTORS 

        """
        super().__init__()

        self.parallel = parallel
        self.N = x_example.numel()
        self.input_shape = x_example.shape
        self.Mat = FlatForward(Mat, self.input_shape)
        self.multi_Mat = MultiApply(self.Mat, self.parallel)
        if Matx_example is None:
            y_example = Mat(x_example)
        else:
            y_example = Matx_example
        self.M = y_example.numel()
        self.output_shape = y_example.shape
        self.MatT = FlatForward(MatT, self.output_shape)
        self.multi_MatT = MultiApply(self.MatT, self.parallel)
        self.device = x_example.device
        self.dtype = x_example.dtype
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.k_sing_vals = k_sing_vals

        self.X_init = None
        self.X_temp = None
        self.set_initial_vectors(X_init)

        del y_example

    def empty_all(self):
        with torch.no_grad():
            self.Mat = None
            self.MatT = None
            self.multi_Mat.erase()
            self.multi_MatT.erase()
            self.X_init = None
            self.X_temp = None
            self.X_final = None
            self.history_ray_quotient = []
            self.history_sing_vals = []
            self.history_time = []
            self.sing_vals_temp = []

        torch.cuda.empty_cache()

    def set_initial_vectors(self, X_init=None):
        with torch.no_grad():
            if X_init == None:
                X_init = generate_rand_ortho_vects(self.N, self.k_sing_vals,
                                                   **(self.factory_kwargs))

            assert X_init.shape == (self.N, self.k_sing_vals)

            self.X_init = X_init

    def tqdm_set_description(self, ray, sing_vals, tqdm_iter, algo_name):
        with torch.no_grad():
            assert isinstance(algo_name, str)

            if isinstance(ray, torch.Tensor):
                ray_val = ray.item()
            else:
                ray_val = ray

            text = ""
            for sigma in sing_vals:
                text += f"{sigma:1.3e}, "

            tqdm_iter.set_description(algo_name +
                                      f"| Ray: {ray_val:1.3e}, Sing: [" + text + "]")

    def initialize_optim_rayleigh(self):
        with torch.no_grad():
            self.history_ray_quotient = []
            self.history_sing_vals = []
            self.history_time = []
            self.sing_vals_temp = []

            self.ray_prev = 1e10
            self.ray_actual = None

    def init_time(self):
        with torch.no_grad():
            self.time0 = time()

    def save_to_history(self, ray_actual, sing_vals):
        with torch.no_grad():
            if isinstance(ray_actual, torch.Tensor):
                ray_val = ray_actual.item()
            else:
                ray_val = ray_actual

            if isinstance(sing_vals, torch.Tensor):
                sing_vals_list = sing_vals.tolist()
            else:
                sing_vals_list = sing_vals

            self.history_ray_quotient.append(ray_val)
            self.history_sing_vals.append(sing_vals_list)
            self.history_time.append(time() - self.time0)

    def stopping_criteria(self, ray_actual, tol, time_max):
        with torch.no_grad():
            # Set ray_actual and ray_prev in float type
            if isinstance(ray_actual, torch.Tensor):
                self.ray_actual = ray_actual.item()
            else:
                self.ray_actual = ray_actual

            # Stopping criteria
            stop_criteria = abs(
                self.ray_actual - self.ray_prev) / (self.ray_prev + 1e-12) < tol
            self.ray_prev = self.ray_actual

            # Add time stopping criteria
            stop_criteria = stop_criteria or (time() - self.time0 > time_max)

            return stop_criteria

    def get_history(self, X, n_vect_to_keep=None):
        """
        Get the history

        Parameters
        ----------
        X : tensor of shape (N, k_sing_vals)
        n_vect_to_keep : int, optional
            The default is k_sing_vals.

        Returns
        -------
        X : tensor of shape (N, k_sing_vals)
            orthogonalized tensor.
        sing_vals : list
            actual singular values.
        hist_ray : list
            history of the rayleight quotient values.
        hist_sing : list
            history of the singular values.
        hist_time : list
            history of the time values.

        """
        # with torch.no_grad():
        #     if n_vect_to_keep is None:
        #         n_vect_to_keep = self.k_sing_vals

        #     X, _ = torch.linalg.qr(X)

        # MatX = self.multi_Mat(X)

        with torch.no_grad():
            # X_H_X = MatX.T @ MatX

            # U, S, _ = torch.linalg.svd(X_H_X)
            # U, sing_vals = U.flip((1,)), S.flip((0,))[:n_vect_to_keep]**0.5
            # X = X @ U[:, :n_vect_to_keep]

            # tensor_empty_cache(U, X_H_X, MatX, S)
            # del U, X_H_X, MatX, S

            (hist_ray, hist_sing, hist_time) = (self.history_ray_quotient,
                                                self.history_sing_vals,
                                                self.history_time)

            return (X, hist_ray, hist_sing, hist_time)

    def save_experiment(self, filename, X, sing_vals, hist_ray, hist_sing, hist_time):
        with torch.no_grad():
            torch.save((X, sing_vals, hist_ray, hist_sing, hist_time), filename)

    def load_experiment(self, filename):
        return sing_solver_load_experiment(filename, **self.factory_kwargs)

    def plot_history(self, hist_ray=None, hist_sing=None, hist_time=None):
        with torch.no_grad():
            if hist_ray is None:
                hist_ray = self.history_ray_quotient
            if hist_sing is None:
                hist_sing = self.history_sing_vals
            if hist_time is None:
                hist_time = self.history_time

            sing_solver_plot_history(hist_ray, hist_sing, hist_time)

    def lobpcg(self, n_step=100, X_init=None, tol=1e-5, time_max=1e10,
               precond_fn=None, largest=False, method="ortho"):

        self.initialize_optim_rayleigh()

        def single_apply_A(x): return self.MatT(
            self.Mat(x.view(self.input_shape))).ravel()
        apply_A = MultiApply(single_apply_A, self.parallel)
        A_shape = (self.N, self.N)
        A_device = self.device
        A_dtype = self.dtype

        def single_apply_sub_A(x): return self.Mat(
            x.view(self.input_shape)).ravel()
        apply_sub_A = MultiApply(single_apply_sub_A, self.parallel)
        sub_dim_A = self.M

        t = tqdm(total=n_step)

        def tracker(lobpcg_worker):
            if lobpcg_worker.ivars["istep"] > 1:
                sing_vals = lobpcg_worker.E**0.5
                ray_actual = torch.sum(sing_vals**2)
                self.tqdm_set_description(ray_actual, sing_vals, t, "lobpcg")

                if lobpcg_worker.ivars["istep"] % 10 == 0:
                    if self.stopping_criteria(ray_actual, tol, time_max):
                        lobpcg_worker.bvars['force_stop'] = True

            t.update(1)

        self.init_time()

        sing_vals, sing_vects = lobpcg(apply_A, A_shape, A_device, A_dtype,
                                       apply_sub_A,
                                       sub_dim_A,
                                       self.k_sing_vals,
                                       None,
                                       self.X_init if X_init == None else X_init,
                                       self.k_sing_vals,
                                       precond_fn,
                                       n_step,
                                       tol,
                                       largest,
                                       method,
                                       tracker,
                                       None,
                                       None,
                                       None,
                                       0.,
                                       True)

        (X, hist_ray, hist_sing, hist_time
         ) = self.get_history(sing_vects)

        tensor_empty_cache(sing_vects)
        del sing_vects

        return (X, sing_vals, hist_ray, hist_sing, hist_time)


if __name__ == "__main__":
    ### SET THE OPERATOR ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    factory_kwargs = {"device": device, "dtype": dtype}

    N = 100
    lin_sp = torch.logspace(-2, -1, N, **factory_kwargs)

    def Phi(x):
        return lin_sp[:, None] * x

    x_example = torch.randn((N, 2), **factory_kwargs)
    k_sing_vals = 5
    X_init = None
    tol = 1e-4
    n_step = 100

    solver = SingularSolver(Phi, Phi, x_example, k_sing_vals=k_sing_vals)

    ### LOBPCG ###
    (X, sing_vals, hist_ray, hist_sing, hist_time) = solver.lobpcg(
        n_step * 100, X_init, tol / 100)
    solver.plot_history()