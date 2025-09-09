#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Pierre Weiss, Nathanael Munier 
"""

import torch
import torch.nn as nn

from pathlib import Path

from tqdm import tqdm
from .singular_solvers import SingularSolver, singular_vectors
from .utils import FlatForward, tensor_empty_cache

"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""




class ModelOperator(nn.Module):
    def __init__(self, Phi, x_example, parallel=False):
        """
        Create a model whose direct operator is defined by the function Phi.
        As precised above, inside this class, Phi is a "flat" operator which
            means that the input and output are flatten tensors.
        
        The main functions of this class are:
            - forward: applying the flatten Phi function.
            - Jacobian related functions (jacobian_compute, get_jacobian, 
                                          set_jacobian_mult_fn, _jvp_fn, _vjp_fn)
            - Singular solver related functions (svd_extract_singular_vectors, 
                                                 singular_pairs_solve)
        
        Parameters
        ----------
        Phi : map
            Direct model function.
        x_example : tensor
            An input example tensor to get the shapes.

        Returns
        -------
        None.

        """
        super().__init__()
        self.N = x_example.numel()
        self.input_shape = x_example.shape
        y_example = Phi(x_example)
        self.M = y_example.numel()
        self.output_shape = y_example.shape
        self.Phi = FlatForward(Phi, self.input_shape)
        self.device = x_example.device
        self.dtype = x_example.dtype
        self.jac = None
        self.singular_values = None
        self.parallel = parallel
        self._vjp_x0 = None
        self._jvp_x0 = None

    def forward(self, *args, **kwargs):
        return self.Phi(*args, **kwargs)

    def _get_dims(self):
        return (self.M, self.N)

    def _get_shapes(self):
        return (self.input_shape, self.output_shape)
    
    def empty_jac(self):
        """
        Empty the memory cache of the Jacobian matrix
        """
        tensor_empty_cache(self._vjp_x0, self._jvp_x0)
        self._vjp_x0, self._jvp_x0 = None, None
    
    def find_singular_pairs(self, x0=None, compute=True, save_result=False,
                            from_svd=True, method="svd_ATA",
                            n_sing_vals=2, save_load_filename=None,
                            sing_steps=1000, sing_thres=0.,
                            precond_fn="Id", time_max=1e10,
                            verbose=False):

        assert ((from_svd and method in ["svd", "svd_ATA"]
                ) or (not (from_svd) and method in ["lobpcg", "jacobi", "lbfgs"]))
        
        if not(isinstance(self.Phi, ModelOperator)):
            self.Phi = ModelOperator(self.Phi, x0)

        # If not already computed and there is no file -> I force the computation and save
        if not(Path(save_load_filename).is_file()) and not(compute):
            print("The singular pairs are not already computed.")
            print(save_load_filename, "is not a file!")
            print("Computation of the singular pairs...")
            compute = True
            save_result = True
        
        # COMPUTE SINGULAR VECTORS #
        if compute:
            assert x0 != None

            if from_svd:
                # COMPUTE THE WHOLE JACOBIAN SINCE IT IS OF SMALL DIMENSION #
                self.jacobian_compute(x0)
                sing_vals, sing_vects = self.svd_extract_singular_vectors(n_sing_vals,
                                                                              largest=False, 
                                                                              method=method)
            else:
                if self.dtype == torch.float32:
                    self.tolerance = 1e-12 * 2
                elif self.dtype == torch.float64:
                    self.tolerance = 1e-24 * 2
                sing_vects, sing_vals, _, _, _ = self.singular_pairs_solve(x0,
                                                                               k=n_sing_vals, 
                                                                               X_init=None,
                                                                               tol=self.tolerance,
                                                                               method=method, 
                                                                               time_max=time_max,
                                                                               verbose=verbose)
            
            sing_vals = torch.abs(sing_vals)
            if save_result:
                self.save_singular_pairs(save_load_filename, sing_vals, sing_vects)
        else:
            sing_vals, sing_vects = self.load_singular_pairs(save_load_filename)

        ### RESHAPE SING VECTS ###
        if sing_vects.ndim == 1:
            sing_vects = sing_vects[:, None]
        assert sing_vects.ndim == 2

        return sing_vals, sing_vects
    
    def jacobian_compute(self, x, method="vmap", verbose=False):
        """
        Compute the flatten jacobian of operator Phi at x.

        Parameters
        ----------
        x : tensor
            Input tensor at which the Jacobian is computed.

        method : string, optional
            Either "vmap", "jvp", "vjp" or "autograd_jacobian". The default is "vmap".

        Returns
        -------
        self.jac: tensor of shape (M, N,)
            The Jacobian matrix
        """

        assert x.numel() == self.N
        assert method in ["vmap", "jvp", "vjp", "autograd_jacobian"]

        self.device = x.device
        self.dtype = x.dtype

        x_fl = x.ravel()

        if method == "autograd_jacobian":
            self.jac = torch.autograd.functional.jacobian(self.Phi, x_fl,
                                                          create_graph=False).detach()
        elif method == "jvp":
            # Compute the jacobian thanks to jvp evaluation on the canonical base
            jac = torch.zeros((self.M, self.N, ),
                              device=self.device, dtype=self.dtype)
            ei = torch.zeros_like(x_fl)
            if verbose:
                range_n = tqdm(range(self.N))
            else:
                range_n = range(self.N)
            for n in tqdm(range_n):
                ei *= 0
                ei[n] = 1
                grad = torch.func.jvp(self.Phi, (x_fl,), (ei,))[1]
                jac[:, n] = grad
            self.jac = jac.detach()

        elif method == "vjp":
            # Compute the jacobian thanks to jvp evaluation on the canonical base
            jac = torch.zeros((self.M, self.N, ),
                              device=self.device, dtype=self.dtype)
            Ax = self.Phi(x)
            ei = torch.zeros_like(Ax)
            if verbose:
                range_m = tqdm(range(self.M))
            else:
                range_m = range(self.M)
            for m in tqdm(range_m):
                ei *= 0
                ei[m] = 1
                grad = torch.func.vjp(self.Phi, x_fl)[1](ei)[0]
                jac[m, :] = grad
            self.jac = jac.detach()

        elif method == "vmap":
            if self.M >= self.N:
                # Parallelizing the jvp evaluation on the canonical base
                eis = torch.eye(self.N, device=self.device, dtype=self.dtype)

                def _grad(ei):
                    return torch.func.jvp(self.Phi, (x_fl,), (ei,))[1]

                jac_fn = torch.vmap(_grad, in_dims=0, randomness="same")
                self.jac = (jac_fn(eis).T).detach()
            else:
                # Parallelizing the vjp evaluation on the canonical base
                eis = torch.eye(self.M, device=self.device, dtype=self.dtype)

                def _grad(ei):
                    _, fun = torch.func.vjp(self.Phi, x_fl)
                    return fun(ei)[0]

                jac_fn = torch.vmap(_grad, in_dims=0, randomness="same")
                self.jac = jac_fn(eis).detach()

        return self.jac
    
    def get_jacobian(self):
        return self.jac

    def _jvp_fn(self, u):
        return torch.func.jvp(self.Phi, (self._jvp_x0,), (u.ravel(),))[1]

    def generate_jvp_fn(self, x0):
        self._jvp_x0 = x0.ravel()

    def _vjp_fn(self, u):
        return self._vjp_aux_fn(u.ravel())[0]

    def generate_vjp_fn(self, x0):
        self._vjp_x0 = x0.ravel()
        self._vjp_aux_fn = torch.func.vjp(self.Phi, self._vjp_x0)[1]

    def set_jacobian_mult_fn(self, x0):
        """
        Precompute the vjp and jvp function at point x0.

        Parameters
        ----------
        x0 : tensor

        Returns
        -------
        None.
        """
        self.generate_jvp_fn(x0)
        self.generate_vjp_fn(x0)

    def svd_extract_singular_vectors(self, k, largest=False, method="svd"):
        """
        Generate k lowest (or largest) singular vectors from the already computed 
            Jacobian of the model operator Phi.
        This function doesn't work if the Jacobian has not been computed 
            or can not be store in memory.
        Please rather use the singular solver (singular_pairs_solve) instead.
        

        Parameters
        ----------
        k : int
            Number of singular vectors to compute.
        largest : bool, optional
            Either largest singular vectors of lowest. The default is False.
        method : string, optional
            Either "svd" or "svd_ATA". The default is "svd".

        Returns
        -------
        sing_vals : tensor of shape (k,)
        sing_vects : tensor of shape (N, k)
        """

        assert self.get_jacobian() != None, "Please compute the Jacobian before (with jacobian_compute) rather use the singular solver (singular_pairs_solve) instead."

        if largest:
            vector_indexes = tuple(range(0, k))
        else:
            vector_indexes = tuple(range(-1, -k-1, -1))

        sing_vects, sing_vals = singular_vectors(self.get_jacobian(),
                                                 vector_indexes=vector_indexes,
                                                 method=method)
        sing_vals = sing_vals.flip(0)
        self.singular_values = sing_vals

        return (sing_vals, sing_vects.T)

    def singular_pairs_solve(self, x0, k, X_init=None, method="lobpcg",
                             tol=1e-10, niter=100000, time_max=1e10,
                             precond_fn="Id", verbose=False):
        """
        Compute the k right singular pairs of the jacobian J_Phi(x0) of the 
            direct model function Phi.
        
        Three main methods are implemented:
            - "lobpcg": Lobpcg algorithm
            - "jacobi": Jacobi algorithm
            - "lbfgs": L-BFGS descent on Rayleigh quotient.
        
        Parameters
        ----------
        x0 : tensor of numel N
            Vecteur at which the jacobian is computed
        k : int, optional
            Number of singular vectors.
        X_init : tensor of numel N * k, or None
            Initial vectors
        method : str, optional
            "lobpcg", "jacobi" or "lbfgs"
        niter : int, optional
            Number of iterations. The default is 100.
        time_max : float, optional
            Maximum time of computation. The default is 1e10.
        precond_fn : matrix function or None or "Id"
            Preconditioned operator. The default is "Id".
        verbose : bool, optional
            Verbose. The default is False.

        Returns
        -------
        (X, sing_vals, hist_ray, hist_sing, hist_time)

        X : tensor of shape (N, k)
            Singular vectors
        sing_vals : tensor of shape k
            Singular values
        hist_ray : list
            History of Rayleigh values
        hist_sing : list
            History of singular values
        hist_time : list
            History of times
        """

        assert method == "lobpcg"
        if not (X_init is None):
            assert X_init.numel() == self.N * k
            X_init = X_init.view((self.N, k))

        if precond_fn == "Id":
            precond_fn = None

        x_fl = x0.ravel()

        def jvp_fn(u):
            return torch.func.jvp(self.Phi, (x_fl,), (u,))[1]

        self.vjp_aux_fn = torch.func.vjp(self.Phi, x_fl)[1]

        def vjp_fn(u):
            return self.vjp_aux_fn(u)[0]

        self.solver = SingularSolver(jvp_fn, vjp_fn, x_fl, k_sing_vals=k,
                                     parallel=self.parallel)

        (X, sing_vals, hist_ray, hist_sing, hist_time
         ) = self.solver.lobpcg(n_step = niter, X_init = X_init, tol = tol, time_max = time_max, largest = False)

        self.solver.empty_all()
        self.solver = None

        return (X, sing_vals, hist_ray, hist_sing, hist_time)
