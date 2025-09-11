# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Nathanaël Munier 
"""

import torch
import torch.nn as nn
import itertools
from math import prod


"""
WARNING : EVERY OPERATOR INCLUDED IN THIS FRAMEWORK WILL BE CONSIDERED AS MATRIX
            AND EACH INPUT AND OUTPUT ARE CONSIDERED AS FLATTEN VECTOR
"""


class Grid(nn.Module):
    def __init__(self, n_points_per_axis = None, grid_length = None, directions = None):
        """
        Initializes a discretized grid on which points of the approximated 
            manifold of the solution set will be estimated.
            
        This grid can be viewed in two manner:
            - The coordinate space grid (in N^D as the grid is of dimension D)
            - The ambient space grid (in R^N as the direction vectors)
        
        Main functionalities are:
            - get_index_generator: a generator of the coordinates 
                of each points of the grid
            - coord_to_z: pass a point $z \in \R^D$ from the coordinate space grid
                to directions @ z $\in \R^N$ in the ambient space grid.
            - Some functions for the graph search tools (_init_bfs, _add_neighbors_of)
            - Some functions to store objects associated to each point of the grid
                (_zero_grid_tensor_create, _tensor_set, _tensor_get)
                
        The coordinate grid ( in R^D) is
        $\prod_{l in lengths} l * [-n_points_per_axis//2, ..., 0, ..., n_points_per_axis//2] $
        The ambient space grid is in R^N.
        directions @ $\prod_{l in lengths} 
                l * [-n_points_per_axis//2, ..., 0, ..., n_points_per_axis//2] $
        
        Parameters
        ----------
        n_points_per_axis : int or list or tuple
            number of discretized point per each direction.
        grid_length : float or list or tuple
            the coordinate space grid goes from -lengths[i] to lengths[i] for direction i.
        directions : tensor of shape (N, D)
            Set of the direction vectors that span the ambient space grid.

        Returns
        -------
        None.
        """
        
        super().__init__()

        if directions != None:
            assert len(directions) >= 2
            D = directions.shape[1]

            if not (type(n_points_per_axis) in [tuple, list]):
                n_points_per_axis = (n_points_per_axis,) * D

            if not (type(grid_length) in [tuple, list]):
                grid_length = (grid_length,) * D

            assert len(n_points_per_axis) == D
            assert len(grid_length) == D
            assert sum([n % 2 == 0 for n in n_points_per_axis]) == 0

            self.D = D
            self.device = directions.device
            self.dtype = directions.dtype
            self.n_points_per_axis = n_points_per_axis
            self.lengths = grid_length
            self.lin_discretized = [torch.linspace(-self.lengths[i], self.lengths[i], n)
                                    for i, n in enumerate(self.n_points_per_axis)]
            self.neigborhood_list = list(itertools.product(*[[-1, 0, 1]
                                                            for n in self.n_points_per_axis]))
            self.directions = directions

    def generate_over_sampled_grid(self, subdivs):
        """
        Add (subdivs-1) regularly distributed points between two points 
            of the initial grid for every dimensions of the grid.

        For instance, if (m,n) is the n_points_per_axis parameter of the initial grid
            then (subdivs * (m-1) + 1, subdivs * (n-1) + 1) is the shape of the new grid.
        
        Example m = 5 points and subdivs = 4:
        The grid is now viewed with subdivs-1 = 3 more points between each previous points
            X       X       X       X       X
            X x x x X x x x X x x x X x x x X
        
        
        Parameters
        ----------
        subdivs : int
            Number of subdivisions to add. (1 will let the grid unchanged)

        Returns
        -------
        over_grid : Grid
            Over sampled grid.

        """
        assert type(subdivs) == int and subdivs >= 1

        new_n_points_per_axis = tuple([(n_dim-1) * subdivs + 1
                                   for n_dim in self.n_points_per_axis])
        over_grid = Grid(n_points_per_axis=new_n_points_per_axis,
                         lengths=self.lengths,
                         directions=self.directions)
        return over_grid

    def get_index_generator(self):
        """
        Iterator of the coordonates of the discrete points of the grid
        """
        return itertools.product(*[range(n_pt) for n_pt in self.n_points_per_axis])

    def _get_sub_index_generator(self, subdivs):
        """
        Iterator of the coordonates of the discrete points of the sub-grid
        """
        return itertools.product(*[range(0, n_pt, subdivs)
                                   for n_pt in self.n_points_per_axis])

    def _init_bfs(self):
        """
        Initialize the Breadth first search
        """
        self.already_in_list = self._zero_grid_tensor_create(dtype=bool,
                                                             device="cpu")
        self.already_computed = self._zero_grid_tensor_create(dtype=bool,
                                                              device="cpu")
        self.n_bfgs_steps = self._zero_grid_tensor_create(dtype=int,
                                                          device="cpu")

    def coord_to_z(self, coordinates):
        """
        Transform a coordinate point (in N^D) to its corresponding ambient point (in R^N)

        Parameters
        ----------
        coordinates : int tuple

        Returns
        -------
        z : tensor of length N

        """
        z = torch.tensor([self.lin_discretized[i_coord][coord]
                          for i_coord, coord in enumerate(coordinates)],
                         device=self.device, dtype=self.dtype)
        return z

    def _add_neighbors_of(self, coordinates):
        """
        Add the neighbors of a point given by the coordinates in the BFS search process.

        Parameters
        ----------
        coordinates : tuple

        Returns
        -------
        new_coordinates : list of tuple
            List of coordinates of the neighbor points.

        """
        new_coordinates = []

        # For all neighbors
        for neigh_add in self.neigborhood_list:
            # Compute potential coordinate and search if it remains in the grid
            neigh_coord = tuple(
                [i + j for i, j in zip(coordinates, neigh_add)])
            in_the_grid = [not (x_neigh >= 0 and x_neigh < n_i)
                           for n_i, x_neigh in zip(self.n_points_per_axis, neigh_coord)]
            criteria_in_the_grid = (sum(in_the_grid) == 0)

            # Inside the grid
            if criteria_in_the_grid:
                # Not already tagged as a neighborhood
                if not (self._tensor_get(neigh_coord, self.already_in_list)):
                    new_coordinates.append(neigh_coord)
                    self._tensor_set(neigh_coord, self.already_in_list, True)
        return new_coordinates

    def _coord_initial_point(self):
        """
        Coordinate of the center point of the grid
        """
        coord_middle = tuple([coord // 2 for coord in self.n_points_per_axis])
        return coord_middle

    def _zero_grid_tensor_create(self, supplement_dims=None, dtype=None,
                                 device=None):
        """
        Create a tensor indexed with the coordinates of the grid in order to store elements.
        

        Parameters
        ----------
        supplement_dims : int tuple, optional
            Supplement dimension of indexation. The default is None.
        dtype : dtype, optional
            The default is None.
        device : device, optional
            The default is None.

        Returns
        -------
        grid_tensor: tensor of shape self.n_points_per_axis + supplement_dims
            Storage tensor linked to the grid.

        """
        if dtype == None:
            dtype = self.dtype
        if device == None:
            device = self.device
        if supplement_dims == None:
            supplement_dims = tuple([])

        return torch.zeros(self.n_points_per_axis + supplement_dims,
                           device=device, dtype=dtype)

    def _tensor_set(self, coordinates, grid_tensor, value):
        """
        Set the value of a storage tensor at position coordinates. 

        Parameters
        ----------
        coordinates : tuple
        grid_tensor : tensor
            Previously defined storage tensor.
        value : int, float, tensor, str...
            Value to store at index coordinates.

        Returns
        -------
        None.

        """
        assert (len(coordinates) <= grid_tensor.ndim)
        if len(coordinates) == grid_tensor.ndim:
            grid_tensor[tuple(coordinates)] = value
        else:
            grid_tensor[tuple(coordinates) + (Ellipsis,)] = value

    def _tensor_get(self, coordinates, grid_tensor):
        """
        Get the value of a storage tensor at position coordinates. 

        Parameters
        ----------
        coordinates : tuple
        grid_tensor : tensor
            Previously defined storage tensor.

        Returns
        -------
        out
            Value at index coordinates.

        """
        if len(coordinates) == grid_tensor.ndim:
            return grid_tensor[tuple(coordinates)].item()
        else:
            return grid_tensor[tuple(coordinates) + (Ellipsis,)]

    def _directions_get(self):
        return self.directions

    def _get_dim(self):
        return self.D

    def _get_n_pts(self):
        return prod(self.n_points_per_axis)
