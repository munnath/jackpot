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

import os


"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""



class Manifold(nn.Module):
    def __init__(self):
        """
        Define a manifold parameterization tool

        Parameters
        ----------

        Returns
        -------
        None.
        """
        super().__init__()

        self.direction_list = None
        self.param_evals = None
        self.param_losses = None
        self.param_is_computed = None
        self.param_optim_steps = None
        self.param_critera_vals = None
        self.param_critera_valid = None
        self.device = None
        self.dtype = None

    def save_results(self, x_est, save_params_filename, grid, experiment_name=""):
        dict_save = {
            "x_est": x_est,
            "param_evals": self.param_evals,
            "param_losses": self.param_losses,
            "param_is_computed": self.param_is_computed,
            "param_optim_steps": self.param_optim_steps,
            "param_criteria_vals": self.param_criteria_vals,
            "param_criteria_valid": self.param_criteria_valid,
            "device": x_est.device,
            "dtype": x_est.dtype,
            "experiment_name": experiment_name,
            "grid_length": grid.lengths,
            "n_points_per_axis": grid.n_points_per_axis,
            "direction_list": grid._directions_get(),
        }
        os.makedirs(os.path.dirname(save_params_filename), exist_ok=True)
        torch.save(dict_save, save_params_filename)

    def load_results(self, load_params_path):
        # Load
        dict_load = torch.load(load_params_path)
        self.device = dict_load["device"]
        self.dtype = dict_load["dtype"]
        self.param_evals = dict_load["param_evals"]
        self.param_losses = dict_load["param_losses"]
        self.param_is_computed = dict_load["param_is_computed"]
        self.param_optim_steps = dict_load["param_optim_steps"]
        self.param_criteria_vals = dict_load["param_criteria_vals"]
        self.param_criteria_valid = dict_load["param_criteria_valid"]
        direction_list = dict_load["direction_list"]
        n_points_per_axis = dict_load["n_points_per_axis"]
        grid_len = dict_load["grid_length"]

        self.experiment_name = dict_load["experiment_name"]

        return direction_list, n_points_per_axis, grid_len

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
    

    def _set_x(self, x):
        self.x = x

    def _set_direction_list(self, direction_list):
        self._is_a_valid_direction_list(direction_list)
        self.direction_list = direction_list

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

    def _empty_cache(self):
        self.direction_list = None
        self.param_evals = None
        self.param_losses = None
        self.param_criteria_valid = None
        self.param_criteria_vals = None
        torch.cuda.empty_cache()



