# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Nathanaël Munier 
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
import os
import re
import torch.optim as optim
from matplotlib import colors
from tqdm import tqdm

from .direct_model import Model
from .additional_criteria import Criteria
from .grid import Grid
from .utils import send_to_cpu

"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""

# %% Set device and data type
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

class Jackpot(nn.Module):
    def __init__(self, Phi, x_est, experiment_name = None, parallel = False,
                 save_rootname = None):
        """
        Initialize Jackpot algorithm

        Parameters
        ----------
        Phi : map from R^N to R^M
            Direct model.
        x_est : tensor of shape (N,)
            Input estimation x*.
        experiment_name : str, optional
            Name of the experiment. The default is "expe_0".
        parallel : bool, optional
            Whether the map Phi is parallelizable. The default is False.
        save_rootname : str or Path
            Directory name where to save data and plots. The default is None.

        Returns
        -------
        None.

        """
        
        super().__init__()
        
        self.device = x_est.device
        self.dtype = x_est.dtype
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # TIMING
        self.jac_spec_timing = None
        self.manifold_timing = None
        
        # EXPERIMENT NAME AND FILENAMES
        self.set_experiment_name(experiment_name, save_rootname)

        # JACOBIAN SINGULAR SPECTRUM VARIABLES
        self.n_singular_pairs = 5
        self.max_compute_time = 3600
        self.load_sing_pairs = False
        self.save_sing_pairs = True
        
        # MANIFOLD COMPUTATION VARIABLES
        self.D = 2
        self.epsilon = 1e-1
        self.n_points_per_axis = 11
        self.grid_length = 1e-2
        self.load_manifold = False
        self.save_manifold = True

        # DISCRETIZED MANIFOLD DATASET
        self.mani_dataset_evals = None
        self.mani_dataset_losses = None
        self.mani_dataset_is_computed = None
        self.mani_dataset_optim_steps = None
        self.mani_dataset_criteria_vals = None
        self.mani_dataset_criteria_valid = None

        if Phi != None and x_est != None:
            self.Phi = Phi
            self.x_est = x_est
            
            # Set the direct model
            self.model = Model(self.Phi, self.x_est, parallel = parallel)

            self.grid = Grid()
            self.sing_vals = None
            self.sing_vects = None

        self.save_plot = True
    
    def __repr__(self):
        return (
            f"Jackpot(\n"
            f"  experiment_name = {self.experiment_name!r},\n"
            f"  input_shape     = {tuple(getattr(self.model, 'input_shape', (0,)))},\n"
            f"  output_shape    = {tuple(getattr(self.model, 'output_shape', (0,)))},\n"
            f"  device          = {self.device},\n"
            f"  dtype           = {self.dtype},\n"
            f"  manifold_dim    = {self.D},\n"
            f"  epsilon         = {self.epsilon}\n"
            f")"
        )

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Jackpot does not support this parameter {k}.")
            setattr(self, k, v)
            if k == "experiment_name":
                self.set_experiment_name(v)

    def jacobian_spectrum(self, **kwargs):
        if self.load_sing_pairs:
            self.jacobian_spectrum_load(**kwargs)
        else:
            self.jacobian_spectrum_compute(**kwargs)
            if self.save_sing_pairs:
                self.jacobian_spectrum_save()


    def manifold(self, **kwargs):
        if self.load_manifold:
            self.manifold_load(**kwargs)
        else:
            self.manifold_compute(**kwargs)
            if self.save_manifold:
                self.manifold_save()

    def get_results(self):
        return self.mani_dataset_evals
    
    def set_experiment_name(self, experiment_name, save_rootname = None):
        # EXPERIMENT NAME
        self.experiment_name = experiment_name
        if save_rootname != None:
            self.save_rootname = Path(save_rootname)
        else:
            self.save_rootname = Path(f"./saves/{self.experiment_name}/")

        # FILENAMES
        self.fname_jac_spect = self.save_rootname / Path(
            f"{self.experiment_name}.singpairs")
        self.fname_jac_spect_plot = self.save_rootname / Path(
            f"{self.experiment_name}_sing_spectrum.png")
        self.fname_manifold = self.save_rootname / Path(
            f"{self.experiment_name}.manifold")
        self.fname_manifold_plot = self.save_rootname / Path(
            f"{self.experiment_name}_discrepancies.png")

    def get_model_dimensions(self):
        return {"input_dim": self.model.N, "output_dim": self.model.M}
    
    def get_jac_singpairs(self):
        return {"sing_vals": self.sing_vals, "sing_vects": self.sing_vects}
    
    def get_grid(self):
        return self.grid
    
    def jacobian_spectrum_compute(self, n_singular_pairs = None, method = None, max_compute_time = None):
        """
        Computing the lowest Jacobian singular pairs (sigma_i, v_i) 
            - sigma_i are the singular values
            - v_i are the right singular vectors

        Parameters
        ----------
        n_singular_pairs : int
            Number of singular pairs to compute.
        method : str or None, optional
            Computational method to get singularpairs. 
            Either "svd" or "svd_ATA" or "lobpcg"
            The default is chosen given the dimensions.
        max_time : int or None, optional
            Maximal time (in seconds) to run the algorithm, 
            it stops whenever this time is up. 
            The default is None which means no time limit.

        Returns
        -------
        None.

        """
        
        # Check that all required input values are given
        assert n_singular_pairs != None or self.n_singular_pairs != None, "The number of singular pairs to compute is required! Please give a value to Jackpot.n_singular_pairs."
        assert max_compute_time != None or self.max_compute_time != None, "The maximal time (in second) to compute singular paris is required! Please give a value to Jackpot.max_compute_time."

        if n_singular_pairs == None:
            n_singular_pairs = self.n_singular_pairs

        if max_compute_time == None:
            max_compute_time = self.max_compute_time
        
        # Main computation
        self.svd_method = method
        if method == None:
            N, M = self.model.N, self.model.M
            
            if N <= 2000:
                if M <= 1000:
                    self.svd_method = "svd"
                else:
                    self.svd_method = "svd_ATA"
            else:
                self.svd_method = "lobpcg"

        try: 
            t1 = time()
            self.sing_vals, self.sing_vects = self.model.find_singular_pairs(compute=True,
                                                x0 = self.x_est.detach(),
                                                save_result = False,
                                                from_svd=(self.svd_method in ["svd", "svd_ATA"]),
                                                method=self.svd_method,
                                                n_singular_pairs=n_singular_pairs,
                                                save_load_filename="",
                                                max_compute_time=max_compute_time)
            self.jac_spec_timing = time() - t1
        except (MemoryError, OverflowError) as e: 
            print(f"Fallback due to memory-related error: {e}") 
            print("Compute using LOBPCG") 
            self.svd_method = "lobpcg"
            
            t1 = time()
            self.sing_vals, self.sing_vects = self.model.find_singular_pairs(compute=True,
                                                x0 = self.x_est,
                                                save_result = False,
                                                from_svd=(self.svd_method in ["svd", "svd_ATA"]),
                                                method=self.svd_method,
                                                n_singular_pairs=n_singular_pairs,
                                                save_load_filename="",
                                                max_compute_time=max_compute_time)
            self.jac_spec_timing = time() - t1
        
        
    
    def compute_parameterization(self, criteria, optim_params,
                                 search_method="bfs", 
                                 log_loss = False,
                                 verbose = False,
                                 save_each_iter = False,
                                 save_file_name = None,
                                 save_experiment_name = None,
                                 early_criteria_stop = False,
                                 smart_initialization = False):
        
        ################################################
        ####### INITIALIZATION #########################
        ################################################
        self.log_loss = log_loss
        self.Phi_x_est = self.Phi(self.x_est)
        (M, N) = self.model._get_dims()

        # Some checks
        assert isinstance(self.grid, Grid)
        assert isinstance(criteria, Criteria)
        assert isinstance(optim_params, dict)
        assert search_method == "bfs"
        assert (optim_params["subdivs"]) >= 1

        ### OVER SAMPLE THE GRID IF NEED ###
        if optim_params["subdivs"] > 1:
            output_grid = self.grid
            act_grid = self.grid.generate_over_sampled_grid(optim_params["subdivs"])
        else:
            output_grid = self.grid
            act_grid = self.grid

        ### Initialize the variable splitting x into AT A x + (Id - AT A x) ####
        # where
        # self.cons_map : A is the least singular vectors in R^{D x N}
        # self.proj_ortho_map : Id - AT A is the projection onto
        # the orthogonal part of the least singular vectors
        self.cons_map = act_grid._directions_get().T
        self.proj_ortho_map = lambda x: x - \
            self.cons_map.T @ (self.cons_map @ x)

        #################### Initialize the storage variables ###################
        self.mani_dataset_evals = act_grid._zero_grid_tensor_create(device="cpu",
                                                             supplement_dims=self.x_est.shape)
        self.mani_dataset_losses = act_grid._zero_grid_tensor_create(device="cpu")
        self.mani_dataset_is_computed = act_grid._zero_grid_tensor_create(device="cpu",
                                                                   dtype=torch.bool)
        self.mani_dataset_optim_steps = act_grid._zero_grid_tensor_create(device="cpu",
                                                                   dtype=torch.int)
        self.mani_dataset_criteria_vals = act_grid._zero_grid_tensor_create(device="cpu",
                                                                     supplement_dims=(criteria.n_criteria(),))
        self.mani_dataset_criteria_valid = act_grid._zero_grid_tensor_create(device="cpu",
                                                                      supplement_dims=(criteria.n_criteria(),), dtype=torch.bool)
        ################################################
        ########### MAINT PART #########################
        ################################################

        if search_method == "bfs":
            ################################################
            ########### INIT THE BFS SEARCH ################
            ################################################
            act_grid._init_bfs()

            ### SPECIAL CASE FOR THE INITIAL POINT ###
            actual_pt_coord = act_grid._coord_initial_point()

            act_grid.already_in_list[tuple(actual_pt_coord)] = True
            x_ortho_init = torch.zeros((N,), **self.factory_kwargs)

            act_grid._tensor_set(actual_pt_coord, self.mani_dataset_is_computed, True)
            act_grid._tensor_set(actual_pt_coord,
                                 self.mani_dataset_evals, send_to_cpu(self.x_est))

            ### ADD NEIGHBOR COORDINATES TO THE LIST ###
            new_coordinates = act_grid._add_neighbors_of(actual_pt_coord)
            initial_losses = [self._eval_loss(act_grid, coord, x_ortho_init)
                              for coord in new_coordinates]
            coords_to_search = [(coord, ini_loss, x_ortho_init.to('cpu'))
                                for coord, ini_loss in 
                                zip(new_coordinates, initial_losses)]

            # all for tqdm #
            self.tqdm_bar = tqdm(total=self.grid._get_n_pts() * optim_params["subdivs"])
            self.tqdm_text = ""
            
            ### BREADTH-FIRST SEARCH (BFS) LOOP ###
            while coords_to_search != []:
                # Get actual coordinates
                ### SELECT NEXT POINT OF THE GRID WITH THE MINIMAL LOSS VALUE ###
                losses = torch.tensor([cts[1] for cts in coords_to_search])
                i_select = torch.argmin(losses)
                actual_pt_coord, _, x_ortho_init = coords_to_search.pop(i_select)
                x_ortho_init = x_ortho_init.to(self.device)
                z = act_grid.coord_to_z(actual_pt_coord)
                
                # tqdm common text
                self.tqdm_text = "param| pos:["
                for ax_pt in actual_pt_coord:
                    self.tqdm_text += f"{ax_pt:d},"
                self.tqdm_text += "]"
                
                ######################################################
                ### MAIN COMPUTATION OF THE PARAMETERIZATION phi(z)
                ######################################################
                phi_z, valid, n_optim = self.compute_local_param(z=z,
                                        x_ortho_init=x_ortho_init,
                                        actual_pt_coord=actual_pt_coord,
                                        criteria=criteria,
                                        optim_params=optim_params,
                                        verbose=verbose,
                                        early_criteria_stop = early_criteria_stop)
                ######################################################
                ######################################################
                
                
                ## SAVE THE RESULT AT POINT z ##
                # Compute the loss
                loss_z = 0.5 * torch.sum((self.Phi(phi_z) - self.Phi(self.x_est))**2)

                act_grid._tensor_set(actual_pt_coord, self.mani_dataset_losses,
                                     send_to_cpu(loss_z))
                act_grid._tensor_set(actual_pt_coord, self.mani_dataset_evals,
                                     send_to_cpu(phi_z))
                act_grid._tensor_set(
                    actual_pt_coord, self.mani_dataset_is_computed, True)
                act_grid._tensor_set(
                    actual_pt_coord, self.mani_dataset_optim_steps, n_optim)
                act_grid.already_computed[actual_pt_coord] = True
                
                
                ############################################
                ### UPDATE THE coords_to_search LIST     ###
                ############################################
                if valid or not (criteria.stop):
                    _, x_ortho = self._x_get_decomposition(phi_z, self.x_est)
                    x_ortho = x_ortho.ravel()
                    
                    ### UPDATE THE INITIAL LOSS AND X_ORTHO VALUES
                    if smart_initialization:
                        for i_coord, (coord, act_loss, _) in enumerate(coords_to_search):
                            # For only the neighbors of the point
                            if torch.sum(torch.tensor(coord) - torch.tensor(actual_pt_coord)) <= 2:
                                comp_loss = self._eval_loss(act_grid, coord, x_ortho_init)
                                # If initializing with the new value x_ortho gets a lowest loss, then change
                                if comp_loss < act_loss:
                                    #print(f"I change this: {comp_loss} < {act_loss}")
                                    coords_to_search[i_coord] = (coord, comp_loss, x_ortho.to('cpu'))
                        
                    ### ADD NEIGHBOR COORDINATES TO THE LIST ###
                    new_coordinates = act_grid._add_neighbors_of(actual_pt_coord)
                    
                    initial_losses = [self._eval_loss(act_grid, coord, x_ortho)
                                      for coord in new_coordinates]
                    
                    coords_to_search = coords_to_search + [(coord, ini_loss, x_ortho.to('cpu'))
                                                           for coord, ini_loss in 
                                                           zip(new_coordinates, initial_losses)]

                self.tqdm_bar.update(1)
                
                if save_each_iter:
                    self._save_manifold(save_file_name, act_grid, experiment_name=save_experiment_name)

            ################################################
            ########### POST COMPUTATION ###################
            ################################################

            # REDUCE THE GRID IF THERE WERE OVER SAMPLING #
            if optim_params["subdivs"] > 1:
                sub_indexes = act_grid._get_sub_index_generator(
                    optim_params["subdivs"])
                sub_select = act_grid._zero_grid_tensor_create(device="cpu",
                                                               dtype=torch.bool)

                for sub_ind in sub_indexes:
                    sub_select[sub_ind] = True

                def sub_sample(x):
                    x = x[sub_select]
                    x = x.view(output_grid.n_points_per_axis + x.shape[1:])
                    return x

                self.mani_dataset_evals = sub_sample(self.mani_dataset_evals)
                self.mani_dataset_losses = sub_sample(self.mani_dataset_losses)
                self.mani_dataset_is_computed = sub_sample(self.mani_dataset_is_computed)
                self.mani_dataset_optim_steps = sub_sample(self.mani_dataset_optim_steps)
                self.mani_dataset_criteria_vals = sub_sample(self.mani_dataset_criteria_vals)
                self.mani_dataset_criteria_valid = sub_sample(
                    self.mani_dataset_criteria_valid)

            grid = output_grid

            # SAVE THE CRITERION VALUES OF EACH PARAMETERIZATION POINTS #
            for index in grid.get_index_generator():
                if grid._tensor_get(index, self.mani_dataset_is_computed):
                    x_ind = grid._tensor_get(index, self.mani_dataset_evals)
                    x_ind = x_ind.to(**self.factory_kwargs)

                    values, evaluates = criteria.evaluate_in_detail(x_ind, self.x_est)
                    grid._tensor_set(index, self.mani_dataset_criteria_vals,
                                     torch.tensor(values, device="cpu"))
                    grid._tensor_set(index, self.mani_dataset_criteria_valid,
                                     torch.tensor(evaluates, dtype=torch.bool, device="cpu"))

    def compute_local_param(self, z, x_ortho_init, actual_pt_coord,
                            criteria, optim_params, verbose=False,
                            early_criteria_stop = False):
        """ Compute the parameterization phi(z) of the manifold
            This is done by solving the following minimization problem:
            phi(z) = argmin_x    sign 1/2 || Ax - A x_est ||² 
            within the constraint Proj (x - x_est) = z
        

        Parameters
        ----------
        z : tensor of shape (D,)
            parameter value.
        x_ortho_init : tensor of shape (N,)
            Initialization of the orthogonal value of x.
        actual_pt_coord : tuple
            coordinate of the constructed point in the grid.
        criteria : Criteria
            stopping criteria.
        optim_params : dict
            optimization parameters.
        verbose : bool, optional
            Show verbose. The default is False.
        early_criteria_stop : bool, optional
            Stop optimization computation whenever all criteria are half verified. 
            The default is False.

        Returns
        -------
        phi_z : tensor of shape (N,)
            manifold point phi(z)
        valid : bool
            Is the point phi(z) in the epsilon-uncertainty region?
        i_optim : int
            number of optimization steps
        """
            
        assert optim_params["method"] in ["gradient_descent", "L-BFGS"]

        (_, N) = self.model._get_dims()

        y0_norm2 = torch.sum(self.Phi_x_est**2)

        (D, N1) = self.cons_map.shape
        assert N1 == N
        assert z.shape == (D,)
        assert x_ortho_init.shape == (N,)
        # Set the initial value
        if x_ortho_init == None:
            # By default, the initial value is taken at the tangent space
            x_ortho_init = torch.zeros((N,), **self.factory_kwargs)

        # Select the loss function
        if self.log_loss:
            flt_loss = self.log_flat_loss
        else:
            flt_loss = self.flat_loss

        # Optimization steps
        if optim_params["method"] == "gradient_descent":

            # PARAMETERS
            f = flt_loss
            i_optim = 0

            x_ortho = x_ortho_init.ravel().clone().detach()
            x_ortho.requires_grad = True
            
            gd = optim.SGD([x_ortho], lr=optim_params["lr"])
            
            with torch.no_grad():
                phi_z = self._x_from_decomposition(self.x_est, z, x_ortho)
                history_gd = []
            while (i_optim < optim_params["max_iter"] and
                         not (criteria.evaluate_half(phi_z, self.x_est))):

                for k in range(optim_params["max_iter_per_step"]):
                    gd.zero_grad()
                    objective = f(z, x_ortho)
                    objective.backward(retain_graph = False)
                    gd.step()
                    with torch.no_grad():
                        if self.log_loss:
                            history_gd.append(torch.exp(objective).item())
                        else:
                            history_gd.append(objective.item())
                with torch.no_grad():                            
                    phi_z = self._x_from_decomposition(self.x_est, z, x_ortho)
                    i_optim += optim_params["max_iter_per_step"]

            if i_optim > 10:
                plt.semilogy(history_gd, label='Gradient descent')
                plt.legend()
                plt.show()
            
            with torch.no_grad():
                phi_z = self._x_from_decomposition(self.x_est, z, x_ortho)

            # Empty GD memory cache
            gd.zero_grad()
            x_ortho.requires_grad = False
            
        elif optim_params["method"] == "L-BFGS":

            # PARAMETERS
            f = flt_loss
            i_optim = 0
            bfgs_cv = False

            history_lbfgs = []

            # L-BFGS STEPS
            def closure():
                lbfgs.zero_grad()
                objective = f(z, x_ortho)
                objective.backward(retain_graph=True)
                snr = 10 * (torch.log10(y0_norm2) - torch.log10(2 * objective))
                
                self.tqdm_bar.set_description(self.tqdm_text + f", \
iter: {i_optim}, loss: {objective.item():.3e}, snr: {snr:.3f}, grad: {x_ortho.grad.norm():.3e}")
                
                return objective

            x_ortho = x_ortho_init.ravel().clone().detach()
            x_ortho.requires_grad = True

            lbfgs = optim.LBFGS([x_ortho],
                                lr=optim_params["lr"],
                                tolerance_grad=optim_params["tol_grad"],
                                tolerance_change=optim_params["tol_change"],
                                history_size=optim_params["history_size"],
                                max_iter=optim_params["max_iter_per_step"],
                                line_search_fn=optim_params["line_search"])
            
            with torch.no_grad():
                phi_z = self._x_from_decomposition(self.x_est, z, x_ortho)
                prev_loss = float('inf')
                history_lbfgs = []

            for i_optim in range(optim_params["max_iter"]):
                loss = lbfgs.step(closure)
                
                with torch.no_grad():
                    if self.log_loss:
                        history_lbfgs.append(torch.exp(loss).item())
                    else:
                        history_lbfgs.append(loss.item())
                
                    phi_z = self._x_from_decomposition(self.x_est, z, x_ortho)

                    # bfgs stops as soon as either
                    #   - the point is in the uncertainty region (given by criteria)
                    #   - the bfgs steps have converged
                    #   - the number of iterations are above the maximum allowed
    
                    if abs(loss.item() - prev_loss) < optim_params["tol_change"]:
                        if verbose:
                            print(
                                "Stopping criterion: Objective not changing significantly.")
                        bfgs_cv = True
                        break
    
                    if x_ortho.grad.norm() < optim_params["tol_grad"]:
                        if verbose:
                            print("Stopping criterion: Gradient norm is small.")
                        bfgs_cv = True
                        break
                    
                    if early_criteria_stop:
                        if criteria.evaluate_half(phi_z, self.x_est):
                            if verbose:
                                print(
                                    "Stopping criterion: The point is in the uncertainty region with eps/2.")
                            break
    
                    prev_loss = loss.item()
    
                    if verbose:
                        print(
                            f"--- {i_optim}, loss:{loss.item():1.3e} criteria:{criteria.evaluate(phi_z, self.x_est)}, half_crit:{criteria.evaluate_half(phi_z, self.x_est)}")
        
            
            
            if i_optim > 10 and verbose:
                plt.semilogy(history_lbfgs, label='L-BFGS')
                plt.legend()
                plt.title(f"bfgs converges: {bfgs_cv}")
                plt.show()

            # Empty lbfgs memory cache !
            lbfgs.zero_grad()
            x_ortho.requires_grad = False
        
        valid = criteria.evaluate(phi_z, self.x_est, verbose=verbose)

        return phi_z, valid, i_optim
    
    def _eval_loss(self, grid, coord, x_ortho):
        return self.flat_loss(grid.coord_to_z(coord), x_ortho).item()
    
    def flat_loss(self, z, x_ortho):
        """ loss function of the parameterization manifold
            sign 1/2 || Phi x_under_constraints - Phi x_est ||²
        # ------------------------------------
        z: vector of shape (D,)
        x_ortho: vector of shape (N,)
        # ------------------------------------
        flat_loss: map R^(N-D) -> R 
        # Even if the shape of the input model is not flat, 
            here the loss takes a flatten (ravel) input
        # ------------------------------------
        """
        x_reshaped = self._x_from_decomposition(self.x_est, z, x_ortho)
        loss = 0.5 * torch.sum((self.Phi(x_reshaped) - self.Phi_x_est)**2)
        return loss
    
    def log_flat_loss(self, z, x_ortho):
        """ loss function of the parameterization manifold
            sign 1/2 || Phi x_under_constraints - Phi x_est ||²
        # ------------------------------------
        z: vector of shape (D,)
        x_ortho: vector of shape (N,)
        # ------------------------------------
        flat_loss: map R^(N-D) -> R 
        # Even if the shape of the input model is not flat, 
            here the loss takes a flatten (ravel) input
        # ------------------------------------
        """
        x_reshaped = self._x_from_decomposition(self.x_est, z, x_ortho)
        norm_loss = 0.5 * torch.sum((self.Phi(x_reshaped) - self.Phi_x_est)**2)
        loss = torch.log(norm_loss)
        return loss
    
    def set_grid_from_direction_list(self, direction_list, grid_length, n_points_per_axis):
        grid = Grid(n_points_per_axis=n_points_per_axis, grid_length=grid_length,
                    directions=direction_list)
        grid._init_bfs()
        return grid
    

    # def _set_direction_list(self, direction_list):
    #     self._is_a_valid_direction_list(direction_list)
    #     self.direction_list = direction_list
    
    def plot_singular_values(self, title="", normalized=True):
        """ Plot the pre-computed singular values of the jacobian of Phi"""

        if self.singular_values == None:
            raise ValueError(
                "Please compute the Jacobian of the operator before.")
        plt.figure()
        if normalized:
            plt.semilogy(
                (self.singular_values / self.singular_values[0]).tolist(), '-og')
            if self.kernel_threshold != None:
                thres = self.kernel_threshold
            plt.ylim(
                [(self.singular_values[-1] / self.singular_values[0]).item(), 1])
        else:
            plt.semilogy(self.singular_values.tolist(), '-og')
            if self.kernel_threshold != None:
                thres = self.kernel_threshold * self.singular_values[0]
            plt.ylim([self.singular_values[-1].item(),
                     self.singular_values[0].item()])

        if self.kernel_threshold != None:
            plt.semilogy([0, self.singular_values.shape[0]-1],
                         [thres, thres], '--b')

        plt.title(title)
        plt.show()
    
    def _x_from_decomposition(self, x_est, z, x_ortho):
        """
        Recover x from its decomposition in the orthogonal sum of spaces 
        kernel and orthogonal of kernel
        x - x_est = x_ker + x_ortho
        With x_ker = self.cons_map.T @ z

        Remark: Here x_ortho == self.proj_ortho_map(x_ortho) 
                since it should verify x_ortho = self.proj_ortho_map(x - x_est)

        Parameters
        ----------
        x_est : tensor
        z : tensor of shape (D,)
            coordinate in the kernel
        x_ortho : tensor of shape (N,)
            coordinate in the orthogonal of kernel

        Returns
        -------
        tensor of same shape as x_est
        """
        x = x_est.ravel() + self.cons_map.T @ z + self.proj_ortho_map(x_ortho)
        return x.view(x_est.shape)

    # def _is_a_valid_direction_list(self, direction_list):
    #     with torch.no_grad():
    #         _, N = self.model._get_dims()
    #         sh = direction_list.shape
    #         assert len(sh) == 2
    #         valid_test = (sh[-1] == N)
    #         valid_test = direction_list.shape
    #         assert valid_test

    def _x_get_decomposition(self, x, x0):
        """
        Decompose x as x_est + a sum of term in kernel and the orthogonal of the kernel

        x - x_est = AT A (x-x_est) + (Id - AT A) (x-x_est)

        Here z = A (x-x_est) and x_ortho = (Id - AT A) (x-x_est)

        Parameters
        ----------
        x, x_est : tensor of same shape

        Returns
        -------
        z
            tensor of shape (D,)
        x_ortho
            tensor of same shape as x

        """

        assert (x.shape == x0.shape)

        z = self.cons_map @ ((x - x0).ravel())
        x_ortho = self.proj_ortho_map((x - x0).ravel())
        return z, x_ortho.view(x0.shape)

    def _save_manifold(self, save_params_filename, grid, experiment_name=""):
        dict_save = {
            "x_est": self.x_est,
            "mani_dataset_evals": self.mani_dataset_evals,
            "mani_dataset_losses": self.mani_dataset_losses,
            "mani_dataset_is_computed": self.mani_dataset_is_computed,
            "mani_dataset_optim_steps": self.mani_dataset_optim_steps,
            "mani_dataset_criteria_vals": self.mani_dataset_criteria_vals,
            "mani_dataset_criteria_valid": self.mani_dataset_criteria_valid,
            "device": self.device,
            "dtype": self.dtype,
            "experiment_name": experiment_name,
            "grid_length": grid.lengths,
            "n_points_per_axis": grid.n_points_per_axis,
            "direction_list": grid._directions_get(),
        }
        os.makedirs(os.path.dirname(save_params_filename), exist_ok=True)
        torch.save(dict_save, save_params_filename)
    
    def check_singular_vectors(self, x_est, sing_vals, sing_vects):
        
        n_singular_pairs = sing_vects.shape[-1]
        
        ### CHECK THE SINGULAR VECTORS ###
        for k in range(n_singular_pairs):
            estims = []
            for d in range(-20, -10):
                delta = 2**d
                jac_est = (self.Phi(x_est + delta * (sing_vects[:, k]).view(x_est.shape))
                            - self.Phi(x_est)).norm().item() / delta
                estims.append(jac_est)
                # print(delta, jac_est)
            print(
                f"{k+1}^th sing val | autograd: {sing_vals[k].item():.4e} | finite difference: {min(estims):.4e}")
    
    
    def _auto_jac_spec_savefile(self, n_vals, suffix = ""):
        if n_vals != None:
            root, ext = str(self.fname_jac_spect).rsplit('.', 1)
            filename = Path(f"{root}_n_{n_vals}{suffix}.{ext}")
            return n_vals, filename
        else:
            filename = Path("")
            for i_vals in range(200, 1, -1):
                root, ext = str(self.fname_jac_spect).rsplit('.', 1)
                try_path = Path(f"{root}_n_{i_vals}{suffix}.{ext}")
                if try_path.is_file():
                    n_vals = i_vals
                    filename = try_path
                    return n_vals, filename
            return None, Path("")
        
    
    def jacobian_spectrum_save(self, n_singular_pairs = None, filename = None, 
                          filename_suffix = ""):
        """
        Save the actual Jacobian spectrum

        Parameters
        ----------
        n_singular_pairs : int, optional
            Number of singular pairs to save. The default is None.
        filename : str
            File where to save the actual Jacobian spectrum. Default is None
        filename_suffix : str
            File name suffix to add. Default is ""

        Returns
        -------
        None.

        """
        if n_singular_pairs == None:
            n_singular_pairs = self.n_singular_pairs
        
        self.sing_vals = self.sing_vals[:n_singular_pairs]
        self.sing_vects = self.sing_vects[:, :n_singular_pairs]

        if filename == None:
            _, filename = self._auto_jac_spec_savefile(n_singular_pairs, suffix = filename_suffix)
        self.model.save_singular_pairs(filename, self.sing_vals, self.sing_vects)
    
    def jacobian_spectrum_load(self, n_singular_pairs = None, filename = None, filename_suffix = ""):
        """
        Load already computed Jacobian spectrum

        Parameters
        ----------
        n_singular_pairs : int, optional
            number of singular pairs to load. The default is None.
        filename : str
            name of the saved file. Default None
        filename_suffix : str
            File name suffix to add. Default ""
        Returns
        -------
        None.

        """
        if filename == None:
            (n_singular_pairs, filename) = self._auto_jac_spec_savefile(n_singular_pairs, 
                                                                        suffix = filename_suffix)
        
        if not(Path(filename).is_file()):
            print(f"There is no Jacobian sprectrum saved file here: {filename}.")
            print(f"load_sing_pairs passed to False.")
            self.load_sing_pairs = False
            self.jacobian_spectrum()
        else:
            self.sing_vals, self.sing_vects = self.model.load_singular_pairs(filename)
            n_sing_load = self.sing_vals.numel()
            
            if n_singular_pairs != None:
                assert n_singular_pairs <= n_sing_load
                self.sing_vals = self.sing_vals[:n_singular_pairs]
                self.sing_vects = self.sing_vects[:, :n_singular_pairs]
                
            print(f"{filename} loaded.\n")
    
    def jacobian_spectrum_plot(self, filename = None, scalefact = 0.4):
        """
        Plot the Jacobian spectrum.

        Parameters
        ----------
        filename : str, optional
            name of the file where to save the plot. The default is "".
        scalefact : float, optional
            matplotlib scale factor of the figure. The default is 0.4.

        Returns
        -------
        None.

        """
        
        if filename == None:
            filename = self.fname_jac_spect_plot
        
        if self.sing_vals != None:
            plt.figure(figsize=(8 * scalefact, 6 * scalefact))
            plt.semilogy(self.sing_vals.cpu().numpy()[::-1].tolist(), marker = "+", 
                          color = "k", markersize = 5, linewidth = 1) 
            plt.xlabel("$n$", loc = 'right'); plt.ylabel("$\sigma_n$", loc = 'top')
            
            if self.save_plot:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.show()
        else:
            print("There is no Jacobian singular spectrum to plot. \n Please compute or load it through jacobian_spectrum_compute or jacobian_spectrum_load functions.")
    
    def manifold_compute(self, D = None, epsilon = None, n_points_per_axis = None, grid_length = None,
                                    add_criteria = None, directions = None, 
                                    stop_criteria = False):
        """
        Compute the jackpot manifold of the uncertainty region

        Parameters
        ----------
        D : int
            dimension of the manifold.
        eps : float
            discrepancy threshold.
        n_points_per_axis : int or tuple or list
            number of discretized points of the manifold per dimension.
        lengths : float or tuple or list, optional
            lengths of the manifold in each directions. The default is None.
        add_criteria : Criteria or None, optional
            Additive criteria if needed. The default is None.
        directions : tensor of shape (N, D), optional
            Per default, already computed singular vectors are used. 
            Otherwise, vectorsof the same shape are required.
            The default is None.
        stop_criteria : bool, optional
            Whether the algorithm should be stopped whenever the criteria are not valid. The default is False.

        Returns
        -------
        None.

        """
        
        # Check that all required input values are given
        assert D != None or self.D != None, "The dimension of the manifold is required! Please give a value to Jackpot.D."
        assert epsilon != None or self.epsilon != None, "The output discrepancy threshold of the manifold is required. Please give a value to Jackpot.epsilon."
        assert n_points_per_axis != None or self.n_points_per_axis != None, "The number of points in each directions of the tangent grid of the manifold is required! Please give a value to Jackpot.n_points_per_axis."
        assert grid_length != None or self.grid_length != None, "The lenghts of the grid in each directions of the tangent grid of the manifold is required. Please give a value to Jackpot.grid_length."

        if D == None:
            D = self.D

        if epsilon == None:
            epsilon = self.epsilon

        if n_points_per_axis == None:
            n_points_per_axis = self.n_points_per_axis

        if grid_length == None:
            grid_length = self.grid_length

        
        ## Criteria defining the manifold ##
        criteria = Criteria(self.x_est.shape)
        if isinstance(add_criteria, Criteria) and add_criteria != None:
            criteria = add_criteria
        
        ## Optimization parameters for parameterization ##
        optim_method = "L-BFGS"  
        line_search = "strong_wolfe"
        lr = 1e0
        history_size = 10
        subdivs = 1
        max_iter_per_step = 1000
        max_iter = 10

        optim_params = self.optim_parameters(optim_method, max_iter,
                                                 max_iter_per_step, history_size, 
                                                 line_search, lr, subdivs,
                                                 tol_change = 1e-5, tol_grad = 1e-5)
        
        # Set the direction of the tangent space
        N = self.x_est.numel()
        
        if directions == None:
            sing_vects = self.sing_vects.view((N, self.sing_vects.numel() // N))
            directions = sing_vects[:, :D]
        else:
            if directions.shape == (N,) and D == 1:
                directions = directions[:, None]
            assert directions.shape == (N, D), f"The given directions (of shape {tuple(directions.shape)}) have not the right shape that should be ({N}, {D})."
            
        # Set the grid
        if grid_length == None:
            grid_len = tuple((torch.min(epsilon / self.sing_vals[:D], 
                                        torch.ones(D, **self.factory_kwargs))).tolist())
        elif type(grid_length) in (int, float):
            grid_len = (grid_length,) * D
        else:
            assert len(grid_length) == D
            grid_len = grid_length
        self.grid = self.set_grid_from_direction_list(
                direction_list=directions, grid_length = grid_len, 
                n_points_per_axis=n_points_per_axis)
        
        # Compute the parameterization of the adversarial manifold and save the timing
        t1 = time()
        self.compute_parameterization(criteria=criteria, optim_params=optim_params, 
                                      search_method="bfs")
        
        self.manifold_timing = time() - t1
    
    def _auto_manifold_savefile(self, D=None, epsilon=None, n_points_per_axis=None, 
                                grid_length=None, suffix=""):
        
        root, ext = str(self.fname_manifold).rsplit('.', 1)
        base_dir = Path(root).parent
        stem = Path(root).name

        # Case 1: all params are provided → build filename
        if None not in (D, epsilon, n_points_per_axis, grid_length):
            # Format floats compactly
            eps_str = f"{epsilon:.0e}" if epsilon < 1e-2 or epsilon > 1e2 else f"{epsilon:.3f}".rstrip("0").rstrip(".")
            L_str   = f"{grid_length:.0e}" if grid_length < 1e-2 or grid_length > 1e2 else f"{grid_length:.3f}".rstrip("0").rstrip(".")

            filename = Path(
                f"{root}_d_{D}_eps_{eps_str}_n_{n_points_per_axis}_l_{L_str}{suffix}.{ext}"
            )
            return D, epsilon, n_points_per_axis, grid_length, filename

        # Case 2: missing specification → search existing files
        else:
            pattern = re.compile(
                rf"{stem}_d_(\d+)_eps_([0-9eE\.\-]+)_n_(\d+)_l_([0-9eE\.\-]+){suffix}\.{ext}$"
            )

            for file in base_dir.glob(f"{stem}_d_*_eps_*_n_*_l_*{suffix}.{ext}"):
                m = pattern.match(file.name)
                if not m:
                    continue

                D_file       = int(m.group(1))
                epsilon_file = float(m.group(2))
                n_file       = int(m.group(3))
                L_file       = float(m.group(4))
                
                print(D, D_file)
                print(epsilon, epsilon_file)
                print(n_points_per_axis, n_file)
                print(grid_length, L_file)

                # Check only non-None params
                if (D is None or D == D_file) and \
                (epsilon is None or abs(epsilon - epsilon_file) < 1e-12) and \
                (n_points_per_axis is None or n_points_per_axis == n_file) and \
                (grid_length is None or abs(grid_length - L_file) < 1e-12):
                    return D_file, epsilon_file, n_file, L_file, file

            # Nothing found
            return D, epsilon, n_points_per_axis, grid_length, Path("")
    
    def manifold_load(self, D=None, epsilon=None, n_points_per_axis=None, 
                      grid_length=None, filename = None, filename_suffix = ""):
        """
        Load previously computed manifold

        Parameters
        ----------
        D : int
            dimension of the manifold.
            default is None: in that case, search for the lowest existing parameterization dimension
        filename : str
            file name to load. Default None
        filename_suffix : str
            file name suffix to add. Default ""
        Returns
        -------
        None.

        """

        # Check that all required input values are given
        assert D != None or self.D != None, "The dimension of the manifold is required! Please give a value to Jackpot.D."
        assert epsilon != None or self.epsilon != None, "The output discrepancy threshold of the manifold is required. Please give a value to Jackpot.epsilon."
        assert n_points_per_axis != None or self.n_points_per_axis != None, "The number of points in each directions of the tangent grid of the manifold is required! Please give a value to Jackpot.n_points_per_axis."
        assert grid_length != None or self.grid_length != None, "The lenghts of the grid in each directions of the tangent grid of the manifold is required. Please give a value to Jackpot.grid_length."

        if D == None:
            D = self.D

        if epsilon == None:
            epsilon = self.epsilon

        if n_points_per_axis == None:
            n_points_per_axis = self.n_points_per_axis

        if grid_length == None:
            grid_length = self.grid_length

        if filename == None:
            (D, epsilon, n_points_per_axis, 
             grid_length, filename) = self._auto_manifold_savefile(D, epsilon, n_points_per_axis, 
                                                                   grid_length, suffix = filename_suffix)
            
            if filename != Path(""):
                (self.D, self.epsilon, 
                 self.n_points_per_axis, self.grid_length) = (D, epsilon, n_points_per_axis, grid_length)
        else:
            #Add filename suffix
            root, ext = str(filename).rsplit('.', 1)
            filename = Path(f"{root}{filename_suffix}.{ext}")
        
        if not(Path(filename).is_file()):
            print(f"There is no Jackpot manifold saved file here: {filename}.")
            print(f"load_manifold passed to False.")
            self.load_manifold = False
            self.manifold()
        else:
            # Load x_est to get device and dtype
            dict_load = torch.load(filename)
            self.device = dict_load["device"]
            self.dtype = dict_load["dtype"]
            self.factory_kwargs = {"device":self.device, "dtype":self.dtype}

            self.x_est = dict_load["x_est"].to(**self.factory_kwargs)

            self.experiment_name = dict_load["experiment_name"]

            # Set the Manifold Dataset
            self.mani_dataset_evals = dict_load["mani_dataset_evals"]
            self.mani_dataset_losses = dict_load["mani_dataset_losses"]
            self.mani_dataset_is_computed = dict_load["mani_dataset_is_computed"]
            self.mani_dataset_optim_steps = dict_load["mani_dataset_optim_steps"]
            self.mani_dataset_criteria_vals = dict_load["mani_dataset_criteria_vals"]
            self.mani_dataset_criteria_valid = dict_load["mani_dataset_criteria_valid"]
            
            direction_list = dict_load["direction_list"]
            n_points_per_axis = dict_load["n_points_per_axis"]
            grid_length = dict_load["grid_length"]


            # Set the grid
            self.grid = self.set_grid_from_direction_list(
                direction_list=direction_list, grid_length = grid_length, 
                n_points_per_axis=n_points_per_axis)

            print(f"{filename} loaded.")
    
    def manifold_save(self, filename = None, filename_suffix = ""):
        """
        Save actual manifold

        Parameters
        ----------
        filename : str or Path
            file name where to save. Default : None
        filename_suffix : str
            file name suffix to append at the end of the filename. Default : None

        Returns
        -------
        None.

        """
        if filename == None:
            _, _, _, _, filename = self._auto_manifold_savefile(self.D, self.epsilon, self.n_points_per_axis, 
                                                                self.grid_length, suffix = filename_suffix)
        else:
            #Add filename suffix
            root, ext = str(filename).rsplit('.', 1)
            filename = Path(f"{root}{filename_suffix}.{ext}")
        
        self._save_manifold(filename, self.grid, experiment_name=self.experiment_name)

    def optim_parameters(self, method="L-BFGS", max_iter=10,
                         max_iter_per_step=200, history_size=10,
                         line_search="strong_wolfe", lr=1.,
                         subdivs=1, tol_change = 1e-5, tol_grad = 1e-5):
        optim_params = {}

        optim_params["method"] = method
        optim_params["max_iter_per_step"] = max_iter_per_step
        optim_params["max_iter"] = max_iter
        optim_params["line_search"] = line_search
        optim_params["lr"] = lr
        optim_params["subdivs"] = subdivs
        optim_params["history_size"] = history_size
        optim_params["tol_change"] = tol_change
        optim_params["tol_grad"] = tol_grad

        return optim_params

    def plot_criteria(self, valid_count = True):
        sh = self.mani_dataset_criteria_valid.shape
        n_criteria = sh[-1]
        grid_dim = len(sh) - 1
        if grid_dim == 1:  # Grid of dim 1
            for i in range(n_criteria):
                plt.figure()
                if valid_count:
                    plt.plot((self.mani_dataset_criteria_vals[..., i] *
                              self.mani_dataset_criteria_valid[..., i]).tolist())
                else:
                    plt.plot((self.mani_dataset_criteria_vals[..., i]).tolist())
                plt.title(f"criteria {i+1}")
                plt.show()
        elif grid_dim == 2:  # Grid of dim 2
            for i in range(n_criteria):
                plt.figure()
                if valid_count:
                    plt.imshow((self.mani_dataset_criteria_vals[..., i] *
                                self.mani_dataset_criteria_valid[..., i]).tolist())
                else:
                    plt.imshow((self.mani_dataset_criteria_vals[..., i]).tolist())
                plt.colorbar()
                plt.title(f"criteria {i+1}")
                plt.show()
    
    def plot_compare_with_linear(self, grid, x_est, with_linear=True,
                                 output_operator=None, norm_type="L2",
                                 relative=True, semilogy=None):

        if grid._get_dim() == 1:
            # ### LINEARIZATION (WITHOUT OPTIMIZATION)
            if output_operator == None:
                out_op = self.Phi
            else:
                if isinstance(output_operator, Model):
                    out_op = output_operator
                else:
                    out_op = Model(output_operator, self.x_est)

            x_init = x_est.clone().detach()
            x_est_norm = x_init.norm()
            ind_middle = grid._coord_initial_point()
            out_x_est = out_op(x_init)
            out_x_est_norm = out_x_est.norm()  # store to save memory

            assert norm_type in ["L2", "SNR"]

            if norm_type == "L2":
                if relative:
                    def norm_fn(x, x_est, x_est_n): return (x - x_est).norm() / x_est_n
                else:
                    def norm_fn(x, x_est, x_est_n): return (x - x_est).norm()
            elif norm_type == "SNR":
                def norm_fn(x, x_est, x_est_n): return - 10 * (torch.log10((x - x_est).norm())
                                                         - torch.log10(x_est_n))

            if with_linear:
                in_snr_lin = torch.zeros((grid.n_points_per_axis))
                out_snr_lin = torch.zeros((grid.n_points_per_axis))
                for ind in grid.get_index_generator():
                    if self.mani_dataset_is_computed[ind]:
                        evals_i = x_init + (grid.coord_to_z(ind) *
                                            grid.directions[..., 0].view(x_init.shape))
                        x_actual = evals_i
    
                        # Input norm
                        in_snr_lin[ind] = norm_fn(x_actual, x_init, x_est_norm).item()
    
                        # Output norm
                        snr = norm_fn(out_op(x_actual),
                                      out_x_est, out_x_est_norm)
                        
                        out_snr_lin[ind] = snr.item()
                    else:
                        in_snr_lin[ind] = torch.inf
                        out_snr_lin[ind] = torch.inf

                in_snr_lin[ind_middle] = torch.inf
                out_snr_lin[ind_middle] = torch.inf

            # WITH OPTIMIZATION
            evals = self.mani_dataset_evals

            in_snr_list = torch.zeros((grid.n_points_per_axis))
            out_snr_list = torch.zeros((grid.n_points_per_axis))
            for ind in grid.get_index_generator():
                if self.mani_dataset_is_computed[ind]:
                    if ind != grid._coord_initial_point():
                        x_actual = evals[ind + (Ellipsis,)].to(self.device)

                        # Input norm
                        in_snr_list[ind] = norm_fn(
                            x_actual, x_init, x_est_norm).item()

                        # Output norm
                        snr = norm_fn(out_op(x_actual),
                                      out_x_est, out_x_est_norm)
                        out_snr_list[ind] = snr.item()
                else:
                    in_snr_list[ind] = torch.inf
                    out_snr_list[ind] = torch.inf

            in_snr_list[ind_middle] = torch.inf
            out_snr_list[ind_middle] = torch.inf

            if semilogy == None:
                semilogy = (norm_type == "L2")

            if semilogy:
                plot_fn = plt.semilogy
            else:
                plot_fn = plt.plot

            plt.figure()
            if with_linear:
                plot_fn(in_snr_lin.tolist(), "--b", linewidth=2.5)
                plot_fn(out_snr_lin.tolist(), "--g", linewidth=2.5)
            plot_fn(in_snr_list.tolist(), "-b", linewidth=1)
            plot_fn(out_snr_list.tolist(), "-g", linewidth=1)
            if with_linear:
                plt.legend(["Linearized input " + norm_type, "Linearized output " + norm_type,
                            "Input " + norm_type, "Output " + norm_type])
            else:
                plt.legend(
                    ["Linearized input " + norm_type, "Input " + norm_type])
            plt.figure()
            plt.show()

    def plot_losses(self, filename = "", grid = None, in_SNR = False,
                    levels = None, color_levels = None, title = "Output losses",
                    in_figure = True):
        
        grid_dim = self.mani_dataset_losses.ndim
        if in_SNR:
            y0_norm = torch.log10(1e-64 + torch.sum((self.Phi(self.x_est))**2)).to('cpu')
            snr_loss = 10 * (y0_norm - torch.log10(2 * self.mani_dataset_losses))
            snr_loss[snr_loss > 1000] = torch.inf
            norm = None
        else:
            snr_loss = self.mani_dataset_losses
            norm = colors.LogNorm()
        
        if grid_dim == 1:  # Grid of dim 1
            if in_figure:    
                plt.figure()
            plt.plot(snr_loss.tolist())
            if self.save_plot:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.title(title)
            if in_figure:
                plt.show()
        elif grid_dim == 2:  # Grid of dim 2
            if in_figure:    
                plt.figure(figsize=(6, 5) )
            if grid != None:
                l1, l2 = grid.lengths
                extend = [-l1,l1,-l2,l2]
            else:
                l = 1
                extend = [-l,l,-l,l]
                
            if norm == None:
                im = plt.imshow(snr_loss.detach().numpy().T[::-1,:], cmap = 'copper',
                                extent=extend, aspect = "auto")
            else:
                im = plt.imshow(snr_loss.detach().numpy().T[::-1,:], cmap = 'copper',
                                norm = norm, extent=extend, aspect = "auto")
            cbar = plt.colorbar(im)
            plt.xlabel("$z_1$")
            plt.ylabel("$z_2$")
            if levels != None and grid != None:
                if grid != None:
                    l1, l2 = grid.lengths
                else:
                    l1, l2 = 1, 1
                if color_levels == None:
                    color_levels = ['g'] * len(levels)
                X = torch.linspace(-l1, l1, len(snr_loss))
                Y = torch.linspace(-l2, l2, len(snr_loss))
                Y, X = torch.meshgrid(X, Y, indexing = "ij")
                snr_loss = torch.nan_to_num(snr_loss, posinf = max(levels)+1)
                snr_loss[grid.n_points_per_axis[0]//2, grid.n_points_per_axis[1]//2] = + torch.inf
                plt.contour(X, Y, snr_loss.detach().numpy().T, levels = levels, 
                            colors = color_levels)
                for level, color in zip(levels, color_levels):
                    cbar.ax.hlines(level, 0, 1, colors=color, linewidth=2)
            
            if self.save_plot:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.title(title)
            if in_figure:
                plt.show()

    def plot_fn_evals(self, fn, grid, title="", device=None):
        grid_dim = self.mani_dataset_losses.ndim

        if (fn == "Id") or (fn is None):
            def fn(x): return x

        y = fn(self.x_est)

        if device == None:
            device = y.device

        fn_evals = grid._zero_grid_tensor_create(device=device, dtype=y.dtype,
                                                 supplement_dims=y.shape)

        for ind in grid.get_index_generator():
            if self.mani_dataset_is_computed[ind]:

                xi = self.mani_dataset_evals[ind + (Ellipsis,)].to(**self.factory_kwargs).view(self.Phi.input_shape)

                grid._tensor_set(ind, fn_evals, (fn(xi)).to(device=device))

        if grid_dim == 1:  # Grid of dim 1
            plt.figure()
            plt.plot(fn_evals.tolist())
            plt.title(title)
            plt.show()
        elif grid_dim == 2:  # Grid of dim 2
            plt.figure()
            plt.imshow(fn_evals.tolist())
            plt.colorbar()
            plt.title(title)
            plt.show()

        return fn_evals

    def plot_discrepancy(self, filename = None, in_SNR = False,
                    levels = None, color_levels = None, title = None):
        """
        Plot || Phi(x) - Phi(x*) || for all x of the adversarial manifold

        Parameters
        ----------
        filename : str, optional
            file name where to save the plot. The default is "".
        in_SNR : bool, optional
            whether colorbar should be plot in SNR or not. The default is False.
        levels : list or tuple, optional
            level line values. The default is None.
        color_levels : list or tuple of colors, optional
            level line colors. The default is None.
        title : str, optional
            title of the plot. The default is None.

        Returns
        -------
        None.

        """
        if filename == None:
            filename = self.fname_manifold_plot
        
        if self.mani_dataset_losses != None:
            self.plot_losses(filename = filename, grid = self.grid, in_SNR = in_SNR, levels = levels, color_levels = color_levels)
        else:
            print("There is no manifold discrepancy to plot. \n Please compute or load it through manifold function.")
