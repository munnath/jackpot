import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import time

from .adversarial import AdversarialManifold
from .model_operator import ModelOperator
from .stop_criteria import StopCriteria


"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""

# %% Set device and data type
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

class Jackpot(nn.Module):
    def __init__(self, Phi, x_est, expe_name = None, parallel = False,
                 save_rootname = None):
        """
        Initialize Jackpot algorithm

        Parameters
        ----------
        Phi : map from R^N to R^M
            Direct model.
        x_est : tensor of shape (N,)
            Input estimation x*.
        expe_name : str, optional
            Name of the experiment. The default is "expe_0".
        Phi_parallel : bool, optional
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
        self.am_timing = None
        
        # EXPERIMENT NAME AND FILENAMES
        self.set_expe_name(expe_name, save_rootname)
        
        # LOAD AND SAVE
        self.sing_load = False
        self.sing_save = not(self.sing_load)
        self.adv_mani_load = False
        self.adv_mani_save = not(self.adv_mani_load)
        self.save_plots = True

        # JACOBIAN SINGULAR SPECTRUM VARIABLES
        self.n_sing = None
        self.sing_max_time = None
        
        # ADVERSARIAL MANIFOLD COMPUTATION VARIABLES
        self.n_dim = None
        self.epsilon = None
        self.n_discr_pts = None
        self.grid_lengths = None

        if Phi != None and x_est != None:
            self.Phi = Phi
            self.x_est = x_est
            
            # Set the direct model and the Adversarial Manifold
            self.model = ModelOperator(self.Phi, self.x_est, parallel = parallel)
            self.adv_mani = AdversarialManifold(self.model, **self.factory_kwargs)

            self.grid = None
            self.sing_vals = None
            self.sing_vects = None
    
    def set_expe_name(self, expe_name, save_rootname = None):
        # EXPERIMENT NAME
        self.expe_name = expe_name
        if save_rootname != None:
            self.save_rootname = Path(save_rootname)
        else:
            self.save_rootname = Path(f"./saves/{self.expe_name}/")

        # FILENAMES
        self.fname_jac_spect = self.save_rootname / Path(
            f"{self.expe_name}_singpairs.pth")
        self.fname_jac_spect_plot = self.save_rootname / Path(
            f"{self.expe_name}_sing_spectrum.png")
        self.fname_adv_mani = self.save_rootname / Path(
            f"{self.expe_name}_adv_mani.pth")
        self.fname_adv_mani_plot = self.save_rootname / Path(
            f"{self.expe_name}_discrepancies.png")


    def __str__(self):
        # lines = ["LOPBCG:"]
        # lines += [f"  iparams={self.iparams}"]
        # lines += [f"  fparams={self.fparams}"]
        # lines += [f"  bparams={self.bparams}"]
        # lines += [f"  ivars={self.ivars}"]
        # lines += [f"  fvars={self.fvars}"]
        # lines += [f"  bvars={self.bvars}"]
        # lines += [f"  tvars={self.tvars}"]
        # lines += [f"  apply_A={self.apply_A}"]
        # lines += [f"  apply_sub_A={self.apply_sub_A}"]
        # lines += [f"  B={self.B}"]
        # lines += [f"  iK={self.iK}"]
        # lines += [f"  X={self.X}"]
        # lines += [f"  E={self.E}"]
        # r = ""
        # for line in lines:
        #     r += line + "\n"
        # return r
        pass


    def get_model_dimensions(self):
        return {"input_dim": self.model.N, "output_dim": self.model.M}
    
    def get_jac_singpairs(self):
        return {"sing_vals": self.sing_vals, "sing_vects": self.sing_vects}
    
    def get_grid(self):
        return self.grid
        
    def get_manifold(self):
        return self.adv_mani
    
    def jac_spectrum_compute(self, n_sing = None, method = None, sing_max_time = None):
        """
        Computing the lowest Jacobian singular pairs (sigma_i, v_i) 
            - sigma_i are the singular values
            - v_i are the right singular vectors

        Parameters
        ----------
        n_vals : int
            Number of singularpairs to compute.
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
        assert n_sing != None or self.n_sing != None, "The number of singular pairs to compute  is required! Please give a value to Jackpot.n_sing."
        assert sing_max_time != None or self.sing_max_time != None, "The maximal time (in second) to compute singular paris is required! Please give a value to Jackpot.sing_max_time."

        if n_sing == None:
            n_sing = self.n_sing

        if sing_max_time == None:
            sing_max_time = self.sing_max_time
        
        # Main computation
        if method == None:
            N, M = self.model.N, self.model.M
            
            if N <= 1000:
                if M <= 1000:
                    method = "svd"
                else:
                    method = "svd_ATA"
            else:
                method = "lobpcg"
        
        t1 = time()
        
        # self.sing_vals, self.sing_vects = self.Phi.find_singular_pairs(self.x_est,
        #                                     compute=True,
        #                                     save_result=False,
        #                                     from_svd=(method in ["svd", "svd_ATA"]),
        #                                     method=method,
        #                                     n_sing_vals=n_sing,
        #                                     save_load_filename="",
        #                                     sing_steps=100000,
        #                                     time_max=sing_max_time)
        
        self.sing_vals, self.sing_vects = self.adv_mani.find_singular_pairs(self.x_est,
                                            compute=True,
                                            save_result=False,
                                            from_svd=(method in ["svd", "svd_ATA"]),
                                            method=method,
                                            n_sing_vals=n_sing,
                                            save_load_filename="",
                                            sing_steps=100000,
                                            time_max=sing_max_time)
        
        self.jac_spec_timing = time() - t1
    
    def _auto_jac_spec_savefile(self, n_vals, suffix = ""):
        if n_vals != None:
            root, ext = str(self.fname_jac_spect).rsplit('.', 1)
            filename = Path(f"{root}_n_{n_vals}{suffix}.{ext}")
        else:
            filename = Path("")
            for i_vals in range(200, 1, -1):
                root, ext = str(self.fname_jac_spect).rsplit('.', 1)
                try_path = Path(f"{root}_n_{i_vals}{suffix}.{ext}")
                if try_path.is_file():
                    n_vals = i_vals
                    filename = try_path
                    break
        return n_vals, filename
    
    def jac_spectrum_save(self, n_sing = None, filename = None, 
                          filename_suffix = ""):
        """
        Save the actual Jacobian spectrum

        Parameters
        ----------
        n_vals : int, optional
            Number of singular pairs to save. The default is None.
        filename : str
            File where to save the actual Jacobian spectrum. Default is None
        filename_suffix : str
            File name suffix to add. Default is ""

        Returns
        -------
        None.

        """
        if n_sing == None:
            n_sing = self.n_sing
        
        self.sing_vals = self.sing_vals[:n_sing]
        self.sing_vects = self.sing_vects[:, :n_sing]
        
        if filename == None:
            _, filename = self._auto_jac_spec_savefile(n_sing, suffix = filename_suffix)
        
        
        self.adv_mani.save_singular_pairs(filename, self.sing_vals, self.sing_vects)
    
    def jac_spectrum_load(self, n_vals = None, filename = None, filename_suffix = ""):
        """
        Load already computed Jacobian spectrum

        Parameters
        ----------
        n_vals : int, optional
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
            n_vals, filename = self._auto_jac_spec_savefile(n_vals, suffix = filename_suffix)
        
        if not(Path(filename).is_file()):
            print(f"There is no such file: {filename}.\n")
        else:
            self.sing_vals, self.sing_vects = self.adv_mani.load_singular_pairs(filename)
            n_sing_load = self.sing_vals.numel()
            
            if n_vals != None:
                assert n_vals <= n_sing_load
                self.sing_vals = self.sing_vals[:n_vals]
                self.sing_vects = self.sing_vects[:, :n_vals]
                
            print(f"{filename} loaded.\n")
    
    def jac_spectrum_plot(self, save_plot = False, filename = None, scalefact = 0.4):
        """
        Plot the Jacobian spectrum.

        Parameters
        ----------
        save_plot : bool, optional
            Whether to save the figure or not. The default is False.
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
            
            if save_plot:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.show()
        else:
            print("There is no Jacobian singular spectrum to plot. \n Please compute or load it through jac_spectrum_compute or jac_spectrum_load functions.")
    
    def adv_manifold_compute(self, n_dim = None, epsilon = None, n_discr_pts = None, grid_lengths = None,
                                    add_criteria = None, directions = None, 
                                    stop_criteria = False):
        """
        Compute the adversarial manifold of the uncertainty region

        Parameters
        ----------
        n_dim : int
            dimension of the adversarial manifold.
        eps : float
            discrepancy threshold.
        n_discr_pts : int or tuple or list
            number of discretized points of the manifold per dimension.
        lengths : float or tuple or list, optional
            lengths of the manifold in each directions. The default is None.
        add_criteria : StopCriteria or None, optional
            Additive criteria if needed. The default is None.
        directions : tensor of shape (N, n_dim), optional
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
        assert n_dim != None or self.n_dim != None, "The dimension of the adversarial manifold is required! Please give a value to Jackpot.n_dim."
        assert epsilon != None or self.epsilon != None, "The output discrepancy threshold of the adversarial manifold is required. Please give a value to Jackpot.n_dim."
        assert n_discr_pts != None or self.n_discr_pts != None, "The number of points in each directions of the tangent grid of the adversarial manifold is required! Please give a value to Jackpot.n_discr_pts."
        assert grid_lengths != None or self.grid_lengths != None, "The lenghts of the grid in each directions of the tangent grid of the adversarial manifold is required. Please give a value to Jackpot.grid_lengths."

        if n_dim == None:
            n_dim = self.n_dim

        if epsilon == None:
            epsilon = self.epsilon

        if n_discr_pts == None:
            n_discr_pts = self.n_discr_pts

        if grid_lengths == None:
            grid_lengths = self.grid_lengths

        
        ## Criteria defining the adversarial manifold ##
        criteria = StopCriteria(self.x_est.shape)
        if isinstance(add_criteria, StopCriteria) and add_criteria != None:
            criteria = add_criteria
        
        ## Optimization parameters for parameterization ##
        optim_method = "L-BFGS"  
        line_search = "strong_wolfe"
        lr = 1e0
        history_size = 10
        subdivs = 1
        max_iter_per_step = 1000
        max_iter = 10

        optim_params = self.adv_mani.optim_parameters(optim_method, max_iter,
                                                 max_iter_per_step, history_size, 
                                                 line_search, lr, subdivs,
                                                 tol_change = 1e-5, tol_grad = 1e-5)
        
        # Set the direction of the tangent space
        N = self.x_est.numel()
        
        if directions == None:
            sing_vects = self.sing_vects.view((N, self.sing_vects.numel() // N))
            directions = sing_vects[:, :n_dim]
        else:
            if directions.shape == (N,) and n_dim == 1:
                directions = directions[:, None]
            assert directions.shape == (N, n_dim), f"The given directions (of shape {tuple(directions.shape)}) have not the right shape that should be ({N}, {n_dim})."
            
        # Set the grid
        if grid_lengths == None:
            grid_len = tuple((torch.min(epsilon / self.sing_vals[:n_dim], 
                                        torch.ones(n_dim, **self.factory_kwargs))).tolist())
        elif type(grid_lengths) in (int, float):
            grid_len = (grid_lengths,) * n_dim
        else:
            assert len(grid_lengths) == n_dim
            grid_len = grid_lengths
        self.grid = self.adv_mani.set_grid_from_direction_list(
                direction_list=directions, grid_len = grid_len, 
                n_pts_per_dim=n_discr_pts)
        
        # Compute the parameterization of the adversarial manifold and save the timing
        t1 = time()
        self.adv_mani.compute_parameterization(self.x_est, grid=self.grid, 
                                               criteria=criteria,
                                               optim_params=optim_params,
                                               search_method="bfs")
        
        self.am_timing = time() - t1
    
    def _auto_am_savefile(self, n_dim, suffix = ""):
        if n_dim != None:
            root, ext = str(self.fname_adv_mani).rsplit('.', 1)
            filename = Path(f"{root}_n_{n_dim}{suffix}.{ext}")
        else:
            filename = Path("")
            for i_dim in range(20):
                root, ext = str(self.fname_adv_mani).rsplit('.', 1)
                try_path = Path(f"{root}_n_{i_dim}{suffix}.{ext}")
                if try_path.is_file():
                    n_dim = i_dim
                    filename = try_path
                    break
        return n_dim, filename
    
    def _load_adv_mani_and_grid(self, file_expe_name):
        """
        Load a previoulsy saved model.

        Parameters
        ----------
        file_expe_name : path string
            File where to load the model

        Returns
        -------
        adv_mani : SolutionManifold
            Solution tool with the direct model in it.
        grid : Grid
            discretized grid.
        """
        # Load x0 to get device and dtype
        dict_load = torch.load(file_expe_name)
        device = dict_load["device"]
        dtype = dict_load["dtype"]
        x0 = dict_load["x0"].to(device=device, dtype=dtype)

        # Set the direct model and the adv_mani tool
        model = ModelOperator(self.Phi, x0)
        adv_mani = AdversarialManifold(model, device=device, dtype=dtype)

        # Set the grid
        grid = adv_mani.load_results(file_expe_name)

        return adv_mani, grid
    
    # def set_adv_mani_and_grid(self, x0, sing_vects=None, grid_dim=1,
    #                        dir_len=1e0, n_pts_per_dim=5, tol=None,
    #                        parallel=False):
    #     """
    #     Define a direct model and a grid
    #     If the singular vectors are not already computed (sing_vects = None), 
    #     will only return a adv_mani tool with no grid

    #     Parameters
    #     ----------
    #     Phi : function
    #         direct model.
    #     x0 : tensor of shape (N,)
    #         input tensor.
    #     sing_vects : tensor of shape (N, D), optional
    #         set of singular vectors of the jacobian matrix at x0. The default is None.
    #     grid_dim : int, optional
    #         dimension of the grid. The default is 1.
    #     dir_len : float, optional
    #         length of the grid. The default is 1e0.
    #     n_pts_per_dim : int of tuple of int of length D, optional
    #         number of discretization points on each axis of the grid. The default is 5.

    #     Returns
    #     -------
    #     adv_mani : SolutionManifold
    #         Solution tool with the direct model in it.
    #     grid : Grid
    #         discretized grid.
    #     """

    #     device = x0.device
    #     dtype = x0.dtype

    #     # Set the direct model and the adv_mani tool
    #     N = x0.numel()
    #     model = ModelOperator(self.Phi, x0, parallel=parallel)
    #     adv_mani = AdversarialManifold(model, device=device, dtype=dtype)

    #     # Set the grid (None if no direction are given)
    #     if sing_vects == None:
    #         grid = None
    #     else:
    #         sing_vects = sing_vects.view((N, sing_vects.numel() // N))

    #         grid_len = (dir_len,) * grid_dim
    #         grid = adv_mani.set_grid_from_direction_list(direction_list=sing_vects,
    #                                                      grid_len=grid_len,
    #                                                      n_pts_per_dim=n_pts_per_dim)
    #     return adv_mani, grid
    
    def adv_manifold_load(self, n_dim = None, filename = None, filename_suffix = ""):
        """
        Load previously computed adversarial manifold

        Parameters
        ----------
        n_dim : int
            dimension of the adversarial manifold.
            default is None: in that case, search for the lowest existing parameterization dimension
        filename : str
            file name to load. Default None
        filename_suffix : str
            file name suffix to add. Default ""
        Returns
        -------
        None.

        """
        if filename == None:
            n_dim, filename = self._auto_am_savefile(n_dim, suffix = filename_suffix)
        else:
            #Add filename suffix
            root, ext = str(filename).rsplit('.', 1)
            filename = Path(f"{root}{filename_suffix}.{ext}")
        
        if not(Path(filename).is_file()):
            print(f"There is no such file: {filename}.")
        else:
            self.adv_mani, self.grid = self._load_adv_mani_and_grid(self.Phi, filename)
            print(f"{filename} loaded.")
    
    def adv_manifold_save(self, filename = None, filename_suffix = ""):
        """
        Save actual adversarial manifold

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
            n_dim = self.grid.D
            _, filename = self._auto_am_savefile(n_dim, suffix = filename_suffix)
        else:
            #Add filename suffix
            root, ext = str(filename).rsplit('.', 1)
            filename = Path(f"{root}{filename_suffix}.{ext}")
        
        self.adv_mani.save_results(filename, self.grid, expe_name=self.expe_name)

    def plot_discrepancy(self, save_plot = False, filename = None, in_SNR = False,
                    levels = None, color_levels = None, title = None):
        """
        Plot || Phi(x) - Phi(x*) || for all x of the adversarial manifold

        Parameters
        ----------
        save : bool, optional
            whether to save this plot or not. The default is False.
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
            filename = self.fname_adv_mani_plot
        
        if self.adv_mani.param_losses != None:
            self.adv_mani.plot_losses(save = save_plot, filename = filename, 
                                      grid = self.grid, in_SNR = in_SNR,
                                      levels = levels, color_levels = color_levels)
        else:
            print("There is no adversarial manifold discrepancy to plot. \n Please compute or load it through adv_manifold_compute or adv_manifold_load functions.")