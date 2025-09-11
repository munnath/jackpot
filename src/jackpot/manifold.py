# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Nathanaël Munier 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import colors

from tqdm import tqdm
import os
from pathlib import Path

from .grid import Grid
from .additional_criteria import Criteria
from .direct_model import Model
from .utils import send_to_cpu

"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""



class Manifold(nn.Module):
    def __init__(self, model, device="cuda", dtype=torch.float32):
        """
        Define a manifold parameterization tool

        Parameters
        ----------
        model : Model
            Direct model.
        device : device, optional
            The default is "cuda".
        dtype : dtype, optional
            The default is torch.float32.

        Returns
        -------
        None.
        """
        super().__init__()

        self.x0 = None
        self.Phi = model
        self.Phi_x0 = None
        self.direction_list = None
        self.param_evals = None
        self.param_losses = None
        self.param_is_computed = None
        self.param_optim_steps = None
        self.param_critera_vals = None
        self.param_critera_valid = None
        self.device = device
        self.dtype = dtype
        input_shape, _ = model._get_shapes()
        self.criterion = Criteria(input_shape)
        self.parallel = model.parallel

    def save_results(self, save_params_filename, grid, expe_name=""):
        dict_save = {"x0": self.x0,
                     "param_evals": self.param_evals,
                     "param_losses": self.param_losses,
                     "param_is_computed": self.param_is_computed,
                     "param_optim_steps": self.param_optim_steps,
                     "param_criteria_vals": self.param_criteria_vals,
                     "param_criteria_valid": self.param_criteria_valid,
                     "device": self.device,
                     "dtype": self.dtype,
                     "expe_name": expe_name,
                     "grid_lengths": grid.lengths,
                     "n_pts_per_dim": grid.n_pts_per_dim,
                     "direction_list": grid._directions_get(),
                     }
        os.makedirs(os.path.dirname(save_params_filename), exist_ok=True)
        torch.save(dict_save, save_params_filename)

    def load_results(self, load_params_path):
        # Load
        dict_load = torch.load(load_params_path)
        self.device = dict_load["device"]
        self.dtype = dict_load["dtype"]
        self.x0 = dict_load["x0"].to(device=self.device, dtype=self.dtype)
        self.param_evals = dict_load["param_evals"]
        self.param_losses = dict_load["param_losses"]
        self.param_is_computed = dict_load["param_is_computed"]
        self.param_optim_steps = dict_load["param_optim_steps"]
        self.param_criteria_vals = dict_load["param_criteria_vals"]
        self.param_criteria_valid = dict_load["param_criteria_valid"]
        direction_list = dict_load["direction_list"]
        n_pts_per_dim = dict_load["n_pts_per_dim"]
        grid_len = dict_load["grid_lengths"]

        # Recompute what is needed
        self.Phi_x0 = self.Phi(self.x0)

        self.expe_name = dict_load["expe_name"]

        grid = self.set_grid_from_direction_list(direction_list=direction_list,
                                                 grid_len=grid_len,
                                                 n_pts_per_dim=n_pts_per_dim)
        return grid

    def _get_dims(self):
        return self.Phi._get_dims()

    def _get_shapes(self):
        return self.Phi._get_shapes()

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
    
    

    
    def compute_parameterization(self, x0, grid, criteria, optim_params,
                                 search_method="bfs", 
                                 log_loss = False,
                                 verbose = False,
                                 save_each_iter = False,
                                 save_file_name = None,
                                 save_expe_name = None,
                                 early_criteria_stop = False,
                                 smart_initialization = False):
        
        ################################################
        ####### INITIALIZATION #########################
        ################################################
        self.log_loss = log_loss
        self.x0 = x0
        self.Phi_x0 = self.Phi(x0)
        self.device = x0.device
        self.dtype = x0.dtype
        (M, N) = self._get_dims()

        # Some checks
        assert isinstance(grid, Grid)
        assert isinstance(criteria, Criteria)
        assert isinstance(optim_params, dict)
        assert search_method == "bfs"
        assert (optim_params["subdivs"]) >= 1

        ### OVER SAMPLE THE GRID IF NEED ###
        if optim_params["subdivs"] > 1:
            output_grid = grid
            act_grid = grid.generate_over_sampled_grid(optim_params["subdivs"])
        else:
            output_grid = grid
            act_grid = grid

        ### Initialize the variable splitting x into AT A x + (Id - AT A x) ####
        # where
        # self.cons_map : A is the least singular vectors in R^{D x N}
        # self.proj_ortho_map : Id - AT A is the projection onto
        # the orthogonal part of the least singular vectors
        self.cons_map = act_grid._directions_get().T
        self.proj_ortho_map = lambda x: x - \
            self.cons_map.T @ (self.cons_map @ x)

        #################### Initialize the storage variables ###################
        self.param_evals = act_grid._zero_grid_tensor_create(device="cpu",
                                                             supplement_dims=x0.shape)
        self.param_losses = act_grid._zero_grid_tensor_create(device="cpu")
        self.param_is_computed = act_grid._zero_grid_tensor_create(device="cpu",
                                                                   dtype=torch.bool)
        self.param_optim_steps = act_grid._zero_grid_tensor_create(device="cpu",
                                                                   dtype=torch.int)
        self.param_criteria_vals = act_grid._zero_grid_tensor_create(device="cpu",
                                                                     supplement_dims=(criteria.n_criteria(),))
        self.param_criteria_valid = act_grid._zero_grid_tensor_create(device="cpu",
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
            x_ortho_init = torch.zeros(
                (N,), device=self.device, dtype=self.dtype)

            act_grid._tensor_set(actual_pt_coord, self.param_is_computed, True)
            act_grid._tensor_set(actual_pt_coord,
                                 self.param_evals, send_to_cpu(x0))

            ### ADD NEIGHBOR COORDINATES TO THE LIST ###
            new_coordinates = act_grid._add_neighbors_of(actual_pt_coord)
            initial_losses = [self._eval_loss(act_grid, coord, x_ortho_init)
                              for coord in new_coordinates]
            coords_to_search = [(coord, ini_loss, x_ortho_init.to('cpu'))
                                for coord, ini_loss in 
                                zip(new_coordinates, initial_losses)]

            # all for tqdm #
            self.tqdm_bar = tqdm(total=grid._get_n_pts() * optim_params["subdivs"])
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
                loss_z = 0.5 * torch.sum((self.Phi(phi_z) - self.Phi(x0))**2)

                act_grid._tensor_set(actual_pt_coord, self.param_losses,
                                     send_to_cpu(loss_z))
                act_grid._tensor_set(actual_pt_coord, self.param_evals,
                                     send_to_cpu(phi_z))
                act_grid._tensor_set(
                    actual_pt_coord, self.param_is_computed, True)
                act_grid._tensor_set(
                    actual_pt_coord, self.param_optim_steps, n_optim)
                act_grid.already_computed[actual_pt_coord] = True
                
                
                ############################################
                ### UPDATE THE coords_to_search LIST     ###
                ############################################
                if valid or not (criteria.stop):
                    _, x_ortho = self._x_get_decomposition(phi_z, x0)
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
                    self.save_results(
                        save_file_name, act_grid, expe_name=save_expe_name)

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
                    x = x.view(output_grid.n_pts_per_dim + x.shape[1:])
                    return x

                self.param_evals = sub_sample(self.param_evals)
                self.param_losses = sub_sample(self.param_losses)
                self.param_is_computed = sub_sample(self.param_is_computed)
                self.param_optim_steps = sub_sample(self.param_optim_steps)
                self.param_criteria_vals = sub_sample(self.param_criteria_vals)
                self.param_criteria_valid = sub_sample(
                    self.param_criteria_valid)

            grid = output_grid

            # SAVE THE CRITERION VALUES OF EACH PARAMETERIZATION POINTS #
            for index in grid.get_index_generator():
                if grid._tensor_get(index, self.param_is_computed):
                    x_ind = grid._tensor_get(index, self.param_evals)
                    x_ind = x_ind.to(device=self.device, dtype=self.dtype)

                    values, evaluates = criteria.evaluate_in_detail(x_ind, x0)
                    grid._tensor_set(index, self.param_criteria_vals,
                                     torch.tensor(values, device="cpu"))
                    grid._tensor_set(index, self.param_criteria_valid,
                                     torch.tensor(evaluates, dtype=torch.bool, device="cpu"))

        return None  # self.param_evals


    def compute_local_param(self, z, x_ortho_init, actual_pt_coord,
                            criteria, optim_params, verbose=False,
                            early_criteria_stop = False):
        """ Compute the parameterization phi(z) of the adv_mani manifold
            This is done by solving the following minimization problem:
            phi(z) = argmin_x    sign 1/2 || Ax - Ax0 ||² 
            within the constraint Proj (x - x0) = z
        

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
            Is the point phi(z) in the epsilon-adv_mani set ?
        i_optim : int
            number of optimization steps
        """
            
        assert optim_params["method"] in ["gradient_descent", "L-BFGS"]

        (M, N) = self._get_dims()

        y0_norm2 = torch.sum(self.Phi_x0**2)

        (D, N1) = self.cons_map.shape
        assert N1 == N
        assert z.shape == (D,)
        assert x_ortho_init.shape == (N,)
        # Set the initial value
        if x_ortho_init == None:
            # By default, the initial value is taken at the tangent space
            x_ortho_init = torch.zeros(
                (N,), device=self.device, dtype=self.dtype)

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
                phi_z = self._x_from_decomposition(self.x0, z, x_ortho)
                history_gd = []
            while (i_optim < optim_params["max_iter"] and
                         not (criteria.evaluate_half(phi_z, self.x0))):

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
                    phi_z = self._x_from_decomposition(self.x0, z, x_ortho)
                    i_optim += optim_params["max_iter_per_step"]

            if i_optim > 10:
                plt.semilogy(history_gd, label='Gradient descent')
                plt.legend()
                plt.show()
            
            with torch.no_grad():
                phi_z = self._x_from_decomposition(self.x0, z, x_ortho)

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
                objective.backward(retain_graph=False)
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
                phi_z = self._x_from_decomposition(self.x0, z, x_ortho)
                prev_loss = float('inf')
                history_lbfgs = []

            for i_optim in range(optim_params["max_iter"]):
                loss = lbfgs.step(closure)
                
                with torch.no_grad():
                    if self.log_loss:
                        history_lbfgs.append(torch.exp(loss).item())
                    else:
                        history_lbfgs.append(loss.item())
                
                    phi_z = self._x_from_decomposition(self.x0, z, x_ortho)

                    # bfgs stops as soon as either
                    #   - the point is in the adv_mani set (given by criteria)
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
                        if criteria.evaluate_half(phi_z, self.x0):
                            if verbose:
                                print(
                                    "Stopping criterion: The point is in the adv_mani set with eps/2.")
                            break
    
                    prev_loss = loss.item()
    
                    if verbose:
                        print(
                            f"--- {i_optim}, loss:{loss.item():1.3e} criteria:{criteria.evaluate(phi_z, self.x0)}, half_crit:{criteria.evaluate_half(phi_z, self.x0)}")
        
            
            
            if i_optim > 10 and verbose:
                plt.semilogy(history_lbfgs, label='L-BFGS')
                plt.legend()
                plt.title(f"bfgs converges: {bfgs_cv}")
                plt.show()

            # Empty lbfgs memory cache !
            lbfgs.zero_grad()
            x_ortho.requires_grad = False
        
        valid = criteria.evaluate(phi_z, self.x0, verbose=verbose)

        return phi_z, valid, i_optim
    
    def _eval_loss(self, grid, coord, x_ortho):
        return self.flat_loss(grid.coord_to_z(coord), x_ortho).item()
    
    def flat_loss(self, z, x_ortho):
        """ loss function of the parameterization adv_mani manifold
            sign 1/2 || Phi x_under_constraints - Phi x0 ||²
        # ------------------------------------
        z: vector of shape (D,)
        x_ortho: vector of shape (N,)
        # ------------------------------------
        flat_loss: map R^(N-D) -> R 
        # Even if the shape of the input model is not flat, 
            here the loss takes a flatten (ravel) input
        # ------------------------------------
        """
        x_reshaped = self._x_from_decomposition(self.x0, z, x_ortho)
        loss = 0.5 * torch.sum((self.Phi(x_reshaped) - self.Phi_x0)**2)
        return loss
    
    def log_flat_loss(self, z, x_ortho):
        """ loss function of the parameterization adv_mani manifold
            sign 1/2 || Phi x_under_constraints - Phi x0 ||²
        # ------------------------------------
        z: vector of shape (D,)
        x_ortho: vector of shape (N,)
        # ------------------------------------
        flat_loss: map R^(N-D) -> R 
        # Even if the shape of the input model is not flat, 
            here the loss takes a flatten (ravel) input
        # ------------------------------------
        """
        x_reshaped = self._x_from_decomposition(self.x0, z, x_ortho)
        norm_loss = 0.5 * torch.sum((self.Phi(x_reshaped) - self.Phi_x0)**2)
        loss = torch.log(norm_loss)
        return loss

    def _is_a_valid_direction_list(self, direction_list):
        with torch.no_grad():
            M, N = self.Phi._get_dims()
            sh = direction_list.shape
            assert len(sh) == 2
            valid_test = (sh[-1] == N)
            valid_test = direction_list.shape
            assert valid_test

    def _set_x(self, x):
        self.x = x

    def _set_direction_list(self, direction_list):
        self._is_a_valid_direction_list(direction_list)
        self.direction_list = direction_list

    def set_grid_from_direction_list(self, direction_list, grid_len, n_pts_per_dim):
        grid = Grid(n_pts_per_dim=n_pts_per_dim, lengths=grid_len,
                    directions=direction_list, device=self.device, dtype=self.dtype)
        grid._init_bfs()
        return grid

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

    def _x_from_decomposition(self, x0, z, x_ortho):
        """
        Recover x from its decomposition in the orthogonal sum of spaces 
        kernel and orthogonal of kernel
        x - x0 = x_ker + x_ortho
        With x_ker = self.cons_map.T @ z

        Remark: Here x_ortho == self.proj_ortho_map(x_ortho) 
                since it should verify x_ortho = self.proj_ortho_map(x - x0)

        Parameters
        ----------
        x0 : tensor
        z : tensor of shape (D,)
            coordinate in the kernel
        x_ortho : tensor of shape (N,)
            coordinate in the orthogonal of kernel

        Returns
        -------
        tensor of same shape as x0
        """
        x = x0.ravel() + self.cons_map.T @ z + self.proj_ortho_map(x_ortho)
        return x.view(x0.shape)

    def _x_get_decomposition(self, x, x0):
        """
        Decompose x as x0 + a sum of term in kernel and the orthogonal of the kernel

        x - x0 = AT A (x-x0) + (Id - AT A) (x-x0)

        Here z = A (x-x0) and x_ortho = (Id - AT A) (x-x0)

        Parameters
        ----------
        x, x0 : tensor of same shape

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

    def plot_compare_with_linear(self, grid, x0, with_linear=True,
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
                    out_op = Model(output_operator, self.x0)

            x_init = x0.clone().detach()
            x0_norm = x_init.norm()
            ind_middle = grid._coord_initial_point()
            out_x0 = out_op(x_init)
            out_x0_norm = out_x0.norm()  # store to save memory

            assert norm_type in ["L2", "SNR"]

            if norm_type == "L2":
                if relative:
                    def norm_fn(x, x0, x0_n): return (x - x0).norm() / x0_n
                else:
                    def norm_fn(x, x0, x0_n): return (x - x0).norm()
            elif norm_type == "SNR":
                def norm_fn(x, x0, x0_n): return - 10 * (torch.log10((x - x0).norm())
                                                         - torch.log10(x0_n))

            if with_linear:
                in_snr_lin = torch.zeros((grid.n_pts_per_dim))
                out_snr_lin = torch.zeros((grid.n_pts_per_dim))
                for ind in grid.get_index_generator():
                    if self.param_is_computed[ind]:
                        evals_i = x_init + (grid.coord_to_z(ind) *
                                            grid.directions[..., 0].view(x_init.shape))
                        x_actual = evals_i
    
                        # Input norm
                        in_snr_lin[ind] = norm_fn(x_actual, x_init, x0_norm).item()
    
                        # Output norm
                        snr = norm_fn(out_op(x_actual),
                                      out_x0, out_x0_norm)
                        
                        out_snr_lin[ind] = snr.item()
                    else:
                        in_snr_lin[ind] = torch.inf
                        out_snr_lin[ind] = torch.inf

                in_snr_lin[ind_middle] = torch.inf
                out_snr_lin[ind_middle] = torch.inf

            # WITH OPTIMIZATION
            evals = self.param_evals

            in_snr_list = torch.zeros((grid.n_pts_per_dim))
            out_snr_list = torch.zeros((grid.n_pts_per_dim))
            for ind in grid.get_index_generator():
                if self.param_is_computed[ind]:
                    if ind != grid._coord_initial_point():
                        x_actual = evals[ind + (Ellipsis,)].to(self.device)

                        # Input norm
                        in_snr_list[ind] = norm_fn(
                            x_actual, x_init, x0_norm).item()

                        # Output norm
                        snr = norm_fn(out_op(x_actual),
                                      out_x0, out_x0_norm)
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

    def plot_criteria(self, valid_count = True):
        sh = self.param_criteria_valid.shape
        n_criteria = sh[-1]
        grid_dim = len(sh) - 1
        if grid_dim == 1:  # Grid of dim 1
            for i in range(n_criteria):
                plt.figure()
                if valid_count:
                    plt.plot((self.param_criteria_vals[..., i] *
                              self.param_criteria_valid[..., i]).tolist())
                else:
                    plt.plot((self.param_criteria_vals[..., i]).tolist())
                plt.title(f"criteria {i+1}")
                plt.show()
        elif grid_dim == 2:  # Grid of dim 2
            for i in range(n_criteria):
                plt.figure()
                if valid_count:
                    plt.imshow((self.param_criteria_vals[..., i] *
                                self.param_criteria_valid[..., i]).tolist())
                else:
                    plt.imshow((self.param_criteria_vals[..., i]).tolist())
                plt.colorbar()
                plt.title(f"criteria {i+1}")
                plt.show()

    def plot_losses(self, save = False, filename = "", grid = None, in_SNR = False,
                    levels = None, color_levels = None, title = "Output losses",
                    in_figure = True):
        
        grid_dim = self.param_losses.ndim
        if in_SNR:
            y0_norm = torch.log10(1e-64 + torch.sum((self.Phi(self.x0))**2)).to('cpu')
            snr_loss = 10 * (y0_norm - torch.log10(2 * self.param_losses))
            snr_loss[snr_loss > 1000] = torch.inf
            norm = None
        else:
            snr_loss = self.param_losses
            norm = colors.LogNorm()
        
        if grid_dim == 1:  # Grid of dim 1
            if in_figure:    
                plt.figure()
            plt.plot(snr_loss.tolist())
            if save:
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
                im = plt.imshow(snr_loss.numpy().T[::-1,:], cmap = 'copper',
                                extent=extend, aspect = "auto")
            else:
                im = plt.imshow(snr_loss.numpy().T[::-1,:], cmap = 'copper',
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
                snr_loss[grid.n_pts_per_dim[0]//2, grid.n_pts_per_dim[1]//2] = + torch.inf
                plt.contour(X, Y, snr_loss.numpy().T, levels = levels, 
                            colors = color_levels)
                for level, color in zip(levels, color_levels):
                    cbar.ax.hlines(level, 0, 1, colors=color, linewidth=2)
            
            if save:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.title(title)
            if in_figure:
                plt.show()
    
    
    def plot_fn_evals(self, fn, grid, title="", device=None):
        grid_dim = self.param_losses.ndim

        if (fn == "Id") or (fn is None):
            def fn(x): return x

        y = fn(self.x0)

        if device == None:
            device = y.device

        fn_evals = grid._zero_grid_tensor_create(device=device, dtype=y.dtype,
                                                 supplement_dims=y.shape)

        for ind in grid.get_index_generator():
            if self.param_is_computed[ind]:

                xi = self.param_evals[ind + (Ellipsis,)].to(
                    device=self.device, dtype=self.dtype
                ).view(self.Phi.input_shape)

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

    def _empty_cache(self):
        self.Phi = None
        self.direction_list = None
        self.param_evals = None
        self.param_losses = None
        self.param_criteria_valid = None
        self.param_criteria_vals = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ######### TEMPLATE ##########

    ###############  ALL THE PARAMETERS  #####################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    factory_kwargs = {"device": device, "dtype": dtype}

    verbose = False

    # Load and save parameters
    load_expe = True
    save_expe = not (load_expe)
    expe_name = "tuto"

    # Singular pairs
    n_sing_to_compute = 2
    compute_the_whole_jacobian = True
    recompute_singpairs = True
    save_singpairs = recompute_singpairs
    sing_steps = 10000
    time_max_sing = 5 * 60
    if compute_the_whole_jacobian:
        sing_method = "svd"
    else:
        sing_method = "lobpcg"

    # Solution set optimization parameters
    optim_method = "L-BFGS"  # "gradient_descent"
    line_search = "strong_wolfe"
    lr = 1e0
    history_size = 10
    subdivs = 1
    max_iter_per_step = 1000
    max_iter = 10

    # Grid parameters
    dir_len = 1e0
    n_pts_per_dim = 41

    # Save files
    save_rootname = Path("./saves/")
    os.makedirs(save_rootname, exist_ok=True)
    expe_file = save_rootname / Path(expe_name + ".pth")

    ######################################################################

    ### DIRECT MODEL ###
    N = 100

    def Phi(x): return torch.cat(
        [(x[::2, ...] - x[1::2, ...])**2, x[2::2, ...]], axis=0)
    x_est = torch.randn((N,), device=device, dtype=dtype)

