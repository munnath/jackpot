#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Pierre Weiss, Nathanael Munier 
"""

import torch
import torch.nn as nn

"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""



class StopCriteria(nn.Module):
    def __init__(self, input_shape, stop=True):
        """
        Class to set some stopping criteria defining the adversarial manifold.

        The main functionnalities are:
            - Add a criterion (add_discrepancy_criterion, add_criterion)
            - Evaluation of the criteria (evaluate, evaluate_half, evaluate_in_detail)
        
        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        stop : boolean, optional
            Set if the adv_mani set search computation should stop 
                whenever the criteria are not checked. 
            The default is True.

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.criteria_list = []
        self.input_shape = input_shape
        self.stop = stop

    def n_criteria(self):
        return len(self.criteria_list)

    def add_criterion(self, criterion_fn, threshold, less_than=True,
                      half_factor=0.5):
        """
        Add the criterion defined as criterion_fn(x, x0) < threshold

        Parameters
        ----------
        criterion_fn : function
        threshold : float
        less_than : boolean
        half_factor : float
        """

        def crit_fn(x, x0): return criterion_fn(x.view(self.input_shape),
                                                x0.view(self.input_shape))

        criterion = {"criterion_fn": crit_fn,
                     "threshold": threshold,
                     "less_than": less_than, "half_factor": half_factor}
        self.criteria_list.append(criterion)

    def add_discrepancy_criterion(self, threshold, norm_type="L2",
                                  discr_type="relative", operator="Id",
                                  half_factor=0.5):
        """
        Add a usual discrepancy criterion.

        Parameters
        ----------
        threshold : float
        norm_type : str, optional
            "L1", "L2", "SNR" or "PSNR". The default is "L2".
        discr_type : str, optional
            "relative" or "absolute". The default is "relative".
        operator : None, "Id" or function, optional
            The operator on which the discrepancy is evaluated. The default is "Id".
        half_factor : float, optional
            Threshold with which the optimization steps stop. 
                If criterion_fn(x, x0) < threshold * half_factor then stops.
            The default is 0.5.

        Returns
        -------
        None.
        """
        assert norm_type in ["L1", "L2", "SNR", "PSNR"]
        assert (discr_type in ["relative", "absolute"]) or (
            norm_type in ["SNR", "PSNR"])

        if operator == None or operator == "Id":
            def var_fn(u): return u
        else:
            def var_fn(u): return operator(u.view(self.input_shape))

        if norm_type == "SNR":
            def norm_fn(u): return torch.linalg.norm(u)

            def criterion_fn(x, x0):
                fn_x0 = var_fn(x0)
                return - 10 * (torch.log10(norm_fn(var_fn(x) - fn_x0))
                               - torch.log10(norm_fn(fn_x0)))
            less_than = False
        elif norm_type == "PSNR":
            def mse_fn(x, x0): return torch.sum((x - x0)**2) / x.numel()

            def criterion_fn(x, x0): return (20 * torch.log10(x0.max())
                                             - 10 * torch.log10(mse_fn(x, x0)))
            less_than = False
        else:
            if norm_type == "L2":
                def norm_fn(u): return torch.linalg.norm(u)
            elif norm_type == "L1":
                def norm_fn(u): return torch.sum(torch.abs(u))

            if discr_type == "relative":
                def criterion_fn(x, x0):
                    var_x0 = var_fn(x0)
                    return norm_fn(var_fn(x) - var_x0) / norm_fn(var_x0)
            elif discr_type == "absolute":
                def criterion_fn(x, x0): return norm_fn(var_fn(x) - var_fn(x0))
            less_than = True

        criterion = {"criterion_fn": criterion_fn, "threshold": threshold,
                     "less_than": less_than, "half_factor": half_factor}
        self.criteria_list.append(criterion)

    def reset(self):
        # Reset to an empty criteria list
        self.criteria_list = []

    def evaluate(self, x, x0, half=False, verbose=False):
        """
        Evaluate all the criteria.
        Return True iff all criteria are verified.

        Parameters
        ----------
        x : tensor
            Input to compare.
        x0 : tensor
            Initial value.
        half : bool
            Multiply the thresholds by each criterion half_factors.
            This half criterion determine when to stop the optimization steps.

        Returns
        -------
        criteria_eval: bool
            Validity of all criteria.
        """
        for (i_crit, criterion) in enumerate(self.criteria_list):
            if verbose:
                print(
                    f"Criterion {i_crit+1} of norm: {criterion['criterion_fn'](x, x0).item():.4e}, thres: {criterion['threshold']:.4e}")
            if half:
                thres_factor = criterion["half_factor"]
            else:
                thres_factor = 1

            if criterion["less_than"]:
                if not (criterion["criterion_fn"](x, x0) <= criterion["threshold"] * thres_factor):
                    return False
            else:
                if not (criterion["criterion_fn"](x, x0) >= criterion["threshold"] / thres_factor):
                    return False
        return True

    def evaluate_half(self, x, x0):
        """
        Evaluate all the criteria with thresholds multiplied by half_factor.
        Return True iff all criteria are verified.

        Parameters
        ----------
        x : tensor
            Input to compare.
        x0 : tensor
            Initial value.

        Returns
        -------
        criteria_eval: bool
            Validity of all criteria.
        """
        return self.evaluate(x, x0, half=True, verbose=False)

    def evaluate_in_detail(self, x, x0):
        """
        Evaluate separately all the criterion values (without any 
            comparison with the thresholds).

        Parameters
        ----------
        x : tensor
            Input to compare.
        x0 : tensor
            Initial value.

        Returns
        -------
        values : float list
            The criterion values.
        evals : bool list
            If the criteria is verified for the thresholds or not. 
            List of all criterion values.
        """
        values = [criterion["criterion_fn"](x, x0).item()
                  for criterion in self.criteria_list]

        evals = [(vals <= crit["threshold"]) if (crit["less_than"])
                 else (vals >= crit["threshold"])
                 for (vals, crit) in zip(values, self.criteria_list)]
        return values, evals

