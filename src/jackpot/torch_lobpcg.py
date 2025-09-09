#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Locally Optimal Block Preconditioned Conjugate Gradient methods."""

# https://gitlab.maisondelasimulation.fr/agueroud/pytorch/-/blob/1.6_doc_references/torch/_lobpcg.py
# Author: nmunier
# Inspired from: Pearu Peterson
# Created: March 2024

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm


#%% Import from _linalg_utils :
    
def basis(A):
    """Return orthogonal basis of A columns."""
    return torch.linalg.qr(A).Q

def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)
    
def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def transpose(A):
    """Return transpose of a matrix or batches of matrices."""
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(transpose(X), matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def symeig(A: Tensor, largest: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = torch.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z

__all__ = ["lobpcg"]

#%% Matrix free version of utils functions

def matrix_free_qform(apply_A, S: Tensor, use_sub_A = False, apply_sub_A = None):
    """Return quadratic form :math:`S^T A S`."""
    if not(use_sub_A):
        return matrix_free_bform(S, apply_A, S, use_sub_A)
    else:
        sub_AS = apply_sub_A(S)
        return torch.matmul(transpose(sub_AS), sub_AS)
        

def matrix_free_bform(X: Tensor, apply_A, Y: Tensor, use_sub_A = False, apply_sub_A = None) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    if not(use_sub_A):
        return torch.matmul(transpose(X), apply_A(Y))
    else:
        return torch.matmul(transpose(apply_A(X)), apply_A(Y))

def matrix_free_rayleight(apply_A, X: Tensor, use_sub_A = False, apply_sub_A = None):
    """Compute the Rayleight quotient X^T A X / X^T X"""
    if X.ndim == 1:
        X = X[:, None]
    return torch.diagonal(matrix_free_qform(apply_A = apply_A, S = X, 
                                            use_sub_A = use_sub_A, apply_sub_A = apply_sub_A)
                          ) / torch.diagonal(
                    matrix_free_qform(apply_A = lambda x : x, S = X))
    


#%% LOBPCG FUNCTIONS #########################

class LOBPCGMatrixFreeFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        apply_A,
        A_shape,
        A_device,
        A_dtype,
        k: Optional[int] = None,
        B: Optional[Tensor] = None,
        X: Optional[Tensor] = None,
        n: Optional[int] = None,
        iK: None = None,
        niter: Optional[int] = None,
        tol: Optional[float] = None,
        largest: Optional[bool] = None,
        method: Optional[str] = None,
        tracker: None = None,
        ortho_iparams: Optional[Dict[str, int]] = None,
        ortho_fparams: Optional[Dict[str, float]] = None,
        ortho_bparams: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        D, U = _lobpcg(
            apply_A,
            A_shape,
            A_device,
            A_dtype,
            k,
            B,
            X,
            n,
            iK,
            niter,
            tol,
            largest,
            method,
            tracker,
            ortho_iparams,
            ortho_fparams,
            ortho_bparams,
        )

        ctx.largest = largest

        return D, U

def lobpcg(
    apply_A,
    A_shape,
    A_device,
    A_dtype,
    apply_sub_A: Optional = None,
    sub_dim_A: Optional[int] = None,
    k: Optional[int] = None,
    B: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    n: Optional[int] = None,
    iK: None = None,
    niter: Optional[int] = None,
    tol: Optional[float] = None,
    largest: Optional[bool] = None,
    method: Optional[str] = None,
    tracker: None = None,
    ortho_iparams: Optional[Dict[str, int]] = None,
    ortho_fparams: Optional[Dict[str, float]] = None,
    ortho_bparams: Optional[Dict[str, bool]] = None,
    sing_thres: Optional[float] = 0,
    verbose = False
) -> Tuple[Tensor, Tensor]:
    """Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive definite generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.

    Supported inputs are dense, sparse, and batches of dense matrices.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    .. warning:: The backward method does not support sparse and complex inputs.
      It works only when `B` is not provided (i.e. `B == None`).
      We are actively working on extensions, and the details of
      the algorithms are going to be published promptly.

    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.
      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric
      in first-order optimization routines, prior to running `lobpcg`
      we do the following symmetrization map: `A -> (A + A.t()) / 2`.
      The map is performed only when the `A` requires gradients.

    Args:

      A (Tensor): the input tensor of size :math:`(*, m, m)`
          A is of shape A_shape=(N, N)
      
      A = sub_AT @ sub_A where sub_A is of shape sub_dim_A = (K, N)
          and sub_AT its transposed is of shape (N, K)
      
      A_device : device type ("cpu" or "cuda" for instance)
      
      A_dtype : dtype of the map A (torch.float32 for instance)
      
      sub_dim_A = K is the dimension
      
      B (Tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When not specified, `B` is interpreted as
                  identity matrix.

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, it is used as
                  initial approximation of eigenvectors. X must be a
                  dense tensor.

      iK (function, optional): the input tensor of size :math:`(*, m,
                  m)`. When specified, it will be used as preconditioner.

      k (integer, optional): the number of requested
                  eigenpairs. Default is the number of :math:`X`
                  columns (when specified) or `1`.

      n (integer, optional): if :math:`X` is not specified then `n`
                  specifies the size of the generated random
                  approximation of eigenvectors. Default value for `n`
                  is `k`. If :math:`X` is specified, the value of `n`
                  (when specified) must be the number of :math:`X`
                  columns.

      tol (float, optional): residual tolerance for stopping
                 criterion. Default is `feps ** 0.5` where `feps` is
                 smallest non-zero floating-point number of the given
                 input tensor `A` data type.

      largest (bool, optional): when True, solve the eigenproblem for
                 the largest eigenvalues. Otherwise, solve the
                 eigenproblem for smallest eigenvalues. Default is
                 `True`.

      method (str, optional): select LOBPCG method. See the
                 description of the function above. Default is
                 "ortho".

      niter (int, optional): maximum number of iterations. When
                 reached, the iteration process is hard-stopped and
                 the current approximation of eigenpairs is returned.
                 For infinite iteration but until convergence criteria
                 is met, use `-1`.

      tracker (callable, optional) : a function for tracing the
                 iteration process. When specified, it is called at
                 each iteration step with LOBPCG instance as an
                 argument. The LOBPCG instance holds the full state of
                 the iteration process in the following attributes:

                   `iparams`, `fparams`, `bparams` - dictionaries of
                   integer, float, and boolean valued input
                   parameters, respectively

                   `ivars`, `fvars`, `bvars`, `tvars` - dictionaries
                   of integer, float, boolean, and Tensor valued
                   iteration variables, respectively.

                   `A`, `B`, `iK` - input Tensor arguments.

                   `E`, `X`, `S`, `R` - iteration Tensor variables.

                 For instance:

                   `ivars["istep"]` - the current iteration step
                   `X` - the current approximation of eigenvectors
                   `E` - the current approximation of eigenvalues
                   `R` - the current residual
                   `ivars["converged_count"]` - the current number of converged eigenpairs
                   `tvars["rerr"]` - the current state of convergence criteria

                 Note that when `tracker` stores Tensor objects from
                 the LOBPCG instance, it must make copies of these.

                 If `tracker` sets `bvars["force_stop"] = True`, the
                 iteration process will be hard-stopped.

      ortho_iparams, ortho_fparams, ortho_bparams (dict, optional):
                 various parameters to LOBPCG algorithm when using
                 `method="ortho"`.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    References:

      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
      Preconditioned Eigensolver: Locally Optimal Block Preconditioned
      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
      517-541. (25 pages)
      https://epubs.siam.org/doi/abs/10.1137/S1064827500366124

      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
      Wu. (2002) A Block Orthogonalization Procedure with Constant
      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
      2165-2182. (18 pages)
      https://epubs.siam.org/doi/10.1137/S1064827500370883

      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
      https://epubs.siam.org/doi/abs/10.1137/17M1129830

    """

    return _lobpcg(
        apply_A,
        A_shape,
        A_device,
        A_dtype,
        apply_sub_A,
        sub_dim_A,
        k,
        B,
        X,
        n,
        iK,
        niter,
        tol,
        largest,
        method,
        tracker,
        ortho_iparams,
        ortho_fparams,
        ortho_bparams,
        sing_thres,
        verbose,
    )


def _lobpcg(
    apply_A,
    A_shape,
    A_device,
    A_dtype,
    apply_sub_A: Optional = None,
    sub_dim_A: Optional[int] = None,
    k: Optional[int] = None,
    B: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    n: Optional[int] = None,
    iK: None = None,
    niter: Optional[int] = None,
    tol: Optional[float] = None,
    largest: Optional[bool] = None,
    method: Optional[str] = None,
    tracker: None = None,
    ortho_iparams: Optional[Dict[str, int]] = None,
    ortho_fparams: Optional[Dict[str, float]] = None,
    ortho_bparams: Optional[Dict[str, bool]] = None,
    sing_thres: Optional[float] = 0.,
    verbose : Optional[bool] = False
) -> Tuple[Tensor, Tensor]:
    # A must be square:
    assert A_shape[-2] == A_shape[-1], A_shape
    if B is not None:
        # A and B must have the same shapes:
        assert A_shape == B.shape, (A_shape, B.shape)
    
    ## Assert that each elements of the subdecomposition A = sub_AT @ sub_A are given
    if apply_sub_A == None or sub_dim_A == None:
        apply_sub_A = None
        sub_dim_A = None
        use_sub_A = False
    else:
        use_sub_A = True
    
    dtype = A_dtype
    device = A_device
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps**0.5

    m = A_shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]

    if m < 3 * n:
        raise ValueError(
            f"LPBPCG algorithm is not applicable when the number of A rows (={m})"
            f" is smaller than 3 x the number of requested eigenpairs (={n})"
        )

    method = "ortho" if method is None else method

    iparams = {
        "m": m,
        "n": n,
        "k": k,
        "sub_n": sub_dim_A,
        "niter": 1000 if niter is None else niter,
    }

    fparams = {
        "tol": tol,
    }

    bparams = {"largest": True if largest is None else largest,
               "use_sub_A": use_sub_A}

    if method == "ortho":
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams["ortho_i_max"] = iparams.get("ortho_i_max", 3)
        iparams["ortho_j_max"] = iparams.get("ortho_j_max", 3)
        fparams["ortho_tol"] = fparams.get("ortho_tol", tol)
        fparams["ortho_tol_drop"] = fparams.get("ortho_tol_drop", tol)
        fparams["ortho_tol_replace"] = fparams.get("ortho_tol_replace", tol)
        bparams["ortho_use_drop"] = bparams.get("ortho_use_drop", False)

    if not torch.jit.is_scripting():
        LOBPCGMatrixFree.call_tracker = LOBPCG_call_tracker  # type: ignore[assignment]

    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))

    worker = LOBPCGMatrixFree(apply_A, apply_sub_A, B, X, iK, 
                              iparams, fparams, bparams, method, tracker)

    worker.run(sing_thres = sing_thres, verbose = verbose)

    if not torch.jit.is_scripting():
        LOBPCGMatrixFree.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[assignment]

    return worker.E[:k], worker.X[:, :k]


class LOBPCGMatrixFree:
    """Worker class of LOBPCG methods."""

    def __init__(
        self,
        apply_A,
        apply_sub_A,
        B: Optional[Tensor],
        X: Tensor,
        iK: None,
        iparams: Dict[str, int],
        fparams: Dict[str, float],
        bparams: Dict[str, bool],
        method: str,
        tracker: None,
    ) -> None:
        # constant parameters
        self.apply_A = apply_A
        self.apply_sub_A = apply_sub_A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.tracker = tracker
        m = iparams["m"]
        n = iparams["n"]

        # variable parameters
        self.X = X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)
        self.tvars: Dict[str, Tensor] = {}
        self.ivars: Dict[str, int] = {"istep": 0}
        self.fvars: Dict[str, float] = {"_": 0.0}
        self.bvars: Dict[str, bool] = {"_": False}

    def __str__(self):
        lines = ["LOPBCG:"]
        lines += [f"  iparams={self.iparams}"]
        lines += [f"  fparams={self.fparams}"]
        lines += [f"  bparams={self.bparams}"]
        lines += [f"  ivars={self.ivars}"]
        lines += [f"  fvars={self.fvars}"]
        lines += [f"  bvars={self.bvars}"]
        lines += [f"  tvars={self.tvars}"]
        lines += [f"  apply_A={self.apply_A}"]
        lines += [f"  apply_sub_A={self.apply_sub_A}"]
        lines += [f"  B={self.B}"]
        lines += [f"  iK={self.iK}"]
        lines += [f"  X={self.X}"]
        lines += [f"  E={self.E}"]
        r = ""
        for line in lines:
            r += line + "\n"
        return r

    def update(self):
        """Set and update iteration variables."""
        if self.ivars["istep"] == 0:
            X_norm = float(torch.norm(self.X))
            iX_norm = X_norm**-1
            A_norm = float(torch.norm(self.apply_A(self.X))) * iX_norm
            B_norm = float(torch.norm(matmul(self.B, self.X))) * iX_norm
            self.fvars["X_norm"] = X_norm
            self.fvars["A_norm"] = A_norm
            self.fvars["B_norm"] = B_norm
            self.ivars["iterations_left"] = self.iparams["niter"]
            self.ivars["converged_count"] = 0
            self.ivars["converged_end"] = 0

        if self.method == "ortho":
            self._update_ortho()
        else:
            self._update_basic()

        self.ivars["iterations_left"] = self.ivars["iterations_left"] - 1
        self.ivars["istep"] = self.ivars["istep"] + 1

    def update_residual(self):
        """Update residual R from A, B, X, E."""
        mm = matmul
        self.R = self.apply_A(self.X) - mm(self.B, self.X) * self.E

    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
        # (...) -> int
        prev_count = self.ivars["converged_count"]
        tol = self.fparams["tol"]
        A_norm = self.fvars["A_norm"]
        B_norm = self.fvars["B_norm"]
        E, X, R = self.E, self.X, self.R
        rerr = (
            torch.norm(R, 2, (0,))
            * (torch.norm(X, 2, (0,)) * (A_norm + E[: X.shape[-1]] * B_norm)) ** -1
        )
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                # ignore convergence of following pairs to ensure
                # strict ordering of eigenpairs
                break
            count += 1
        assert (
            count >= prev_count
        ), f"the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease"
        self.ivars["converged_count"] = count
        self.tvars["rerr"] = rerr
        return count

    def stop_iteration(self, sing_thres = 0.):
        """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
        return (
            self.bvars.get("force_stop", False)
            or self.ivars["iterations_left"] == 0
            or self.ivars["converged_count"] >= self.iparams["k"]
            or self.E.min() < sing_thres
        )

    def run(self, sing_thres = 0., verbose = False):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        self.update()

        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()
        
        init_while = True
        
        while not self.stop_iteration(sing_thres = sing_thres * (1-init_while)):
            self.update()
            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()
            
            text = ""
            for sigma in self.E:
                text += f"{sigma**0.5:1.2e}, "
            init_while = False

    @torch.jit.unused
    def call_tracker(self):
        """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
        # do nothing when in TorchScript mode
        pass

    # Internal methods
    
    def mult_by_iK(self, x):
        if self.iK != None:
            return self.iK(x)
        elif type(self.iK) == torch.Tensor:
            return matmul(self.iK, x)
        else:
            return x.clone()
    
    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = qform(matrix_free_qform(apply_A = self.apply_A, S = self.X, 
                                        use_sub_A = self.bparams["use_sub_A"],
                                        apply_sub_A = self.apply_sub_A), Ri)
            E, Z = symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            W = self.mult_by_iK(self.R)
            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W
        else:
            S_ = self.S[:, nc:ns]
            Ri = self._get_rayleigh_ritz_transform(S_)
            M = qform(matrix_free_qform(self.apply_A, S_, self.bparams["use_sub_A"],
                                        self.apply_sub_A), Ri)
            E_, Z = symeig(M, largest)
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, : n - nc]))
            self.E[nc:] = E_[: n - nc]
            P = mm(S_, mm(Ri, Z[:, n : 2 * n - nc]))
            np = P.shape[-1]

            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            self.S[:, n : n + np] = P
            W = self.mult_by_iK(self.R[:, nc:])

            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = qform(matrix_free_qform(self.apply_A, self.X, self.bparams["use_sub_A"],
                                        self.apply_sub_A), Ri)
            E, Z = symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

        else:
            S_ = self.S[:, nc:ns]
            # Rayleigh-Ritz procedure
            E_, Z = symeig(matrix_free_qform(self.apply_A, S_, self.bparams["use_sub_A"],
                                        self.apply_sub_A), largest)

            # Update E, X, P
            self.X[:, nc:] = mm(S_, Z[:, : n - nc])
            self.E[nc:] = E_[: n - nc]
            P = mm(
                S_,
                mm(
                    Z[:, n - nc :],
                    basis(transpose(Z[: n - nc, n - nc :])),
                ),
            )
            np = P.shape[-1]

            # check convergence
            self.update_residual()
            nc = self.update_converged_count()

            # update S
            self.S[:, :n] = self.X
            self.S[:, n : n + np] = P
            W = self._get_ortho(self.R[:, nc:], self.S[:, : n + np])
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.

        .. note:: In the original Rayleight-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::

            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleight-Ritz
          procedure becomes::

            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.

        Args:
        S (Tensor): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.

        Returns:
        Ri (tensor): upper-triangular transformation matrix of size
                     :math:`(n, n)`.

        """
        B = self.B
        SBS = qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** -0.5
        d_col = d_row.reshape(d_row.shape[0], 1)
        # Use torch.linalg.cholesky_solve once it is implemented
        R = torch.linalg.cholesky((SBS * d_row) * d_col, upper=True)
        return torch.linalg.solve_triangular(
            R, d_row.diag_embed(), upper=True, left=False
        )

    def _get_svqb(
        self, U: Tensor, drop: bool, tau: float  # Tensor  # bool  # float
    ) -> Tensor:
        """Return B-orthonormal U.

        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopolousWu2002].

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.

        """
        if torch.numel(U) == 0:
            return U
        UBU = qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)

        # Detect and drop exact zero columns from U. While the test
        # `abs(d) == 0` is unlikely to be True for random data, it is
        # possible to construct input data to lobpcg where it will be
        # True leading to a failure (notice the `d ** -0.5` operation
        # in the original algorithm). To prevent the failure, we drop
        # the exact zero columns here and then continue with the
        # original algorithm below.
        nz = torch.where(abs(d) != 0.0)
        assert len(nz) == 1, nz
        if len(nz[0]) < len(d):
            U = U[:, nz[0]]
            if torch.numel(U) == 0:
                return U
            UBU = qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = torch.where(abs(d) != 0.0)
            assert len(nz[0]) == len(d)

        # The original algorithm 4 from [DuerschPhD2015].
        d_col = (d**-0.5).reshape(d.shape[0], 1)
        DUBUD = (UBU * d_col) * transpose(d_col)
        E, Z = symeig(DUBUD)
        t = tau * abs(E).max()
        if drop:
            keep = torch.where(E > t)
            assert len(keep) == 1, keep
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[(torch.where(E < t))[0]] = t

        return torch.matmul(U * transpose(d_col), Z * E**-0.5)

    def _get_ortho(self, U, V):
        """Return B-orthonormal U with columns are B-orthogonal to V.

        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopolousWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]

        .. note:: If all U columns are B-collinear to V then the
                  returned tensor U will be empty.

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          V (Tensor) : B-orthogonal external basis, size is (m, k)

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
        mm = torch.matmul
        mm_B = matmul
        m = self.iparams["m"]
        tau_ortho = self.fparams["ortho_tol"]
        tau_drop = self.fparams["ortho_tol_drop"]
        tau_replace = self.fparams["ortho_tol_replace"]
        i_max = self.iparams["ortho_i_max"]
        j_max = self.iparams["ortho_j_max"]
        # when use_drop==True, enable dropping U columns that have
        # small contribution to the `span([U, V])`.
        use_drop = self.bparams["ortho_use_drop"]

        # clean up variables from the previous call
        for vkey in list(self.fvars.keys()):
            if vkey.startswith("ortho_") and vkey.endswith("_rerr"):
                self.fvars.pop(vkey)
        self.ivars.pop("ortho_i", 0)
        self.ivars.pop("ortho_j", 0)

        BV_norm = torch.norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(transpose(V), BU)
        i = j = 0
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if torch.numel(U) == 0:
                    # all initial U columns are B-collinear to V
                    self.ivars["ortho_i"] = i
                    self.ivars["ortho_j"] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(transpose(U), BU)
                U_norm = torch.norm(U)
                BU_norm = torch.norm(BU)
                R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
                R_norm = torch.norm(R)
                # https://github.com/pytorch/pytorch/issues/33810 workaround:
                rerr = float(R_norm) * float(BU_norm * U_norm) ** -1
                vkey = f"ortho_UBUmI_rerr[{i}, {j}]"
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(transpose(V), BU)
            VBU_norm = torch.norm(VBU)
            U_norm = torch.norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** -1
            vkey = f"ortho_VBU_rerr[{i}]"
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                # TorchScript needs the class var to be assigned to a local to
                # do optional type refinement
                B = self.B
                assert B is not None
                raise ValueError(
                    "Overdetermined shape of U:"
                    f" #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold"
                )
        self.ivars["ortho_i"] = i
        self.ivars["ortho_j"] = j
        return U


# Calling tracker is separated from LOBPCG definitions because
# TorchScript does not support user-defined callback arguments:
LOBPCG_call_tracker_orig = LOBPCGMatrixFree.call_tracker


def LOBPCG_call_tracker(self):
    self.tracker(self)
    


#%% LOBPCG ESTIMATION ##############

def lobpcg_estim(apply_A, A_shape, A_device = "cuda", A_dtype = torch.float64, 
                 k = 1, cg_n_iter = 10, max_inv_iter = 100, eps = 1e-12, 
                 sing_thres = 1e-2, verbose = False):
    """
    Recursively computing the minimial eigenpairs of the matrix A from apply_A

    Parameters
    ----------
    apply_A : function
        function of matrix multiplication by A.
    A_shape : tuple or int
        (N, N) or N.
    A_dtype : type, optional
        The default is torch.float64.
    A_device : device, optional
        The default is "cuda".
    k : int, optional
        Number of desired eigenpairs. The default is 1.
    cg_n_iter : int, optional
        number of conjugate gradient steps to estimate the inverse. The default is 10.
    max_inv_iter : int, optional
        maximal number of inverse power iterations. The default is 100.
    eps : float, optional
        tolerance threshold. The default is 1e-12.

    Return : Tuple (S, V) of 
            S : eigenvalues sorted from the lowest to the highest
            V : Tensor of shape (N, k) of the associated eigenvectors
    -------
    TYPE
        DESCRIPTION.

    """
    
    if type(A_shape) == int:
        A_shape = (A_shape, A_shape)
    
    eigen = _min_eigenpairs_estim(apply_A, A_shape, A_device, A_dtype, 
                                  k, cg_n_iter, max_inv_iter, eps, 
                                  rayleigh_apply_A = None, sing_thres = sing_thres, 
                                  verbose = verbose)
    
    # A_vects = [apply_A(eigp[0]) for eigp in eigen]
    # vects = torch.cat([eig[None, :] for eig in A_vects], dim = 0)
    # _, S, V = torch.linalg.svd(vects, full_matrices = False)
    # return S.flip(dims = (0,)), V.flip(dims = (0,)).T
    
    vects = torch.cat([eig[0][None, :] for eig in eigen], dim = 0).T
    l = torch.tensor([eig[1] for eig in eigen], dtype = A_dtype, device = A_device)
    
    return l, vects


def _min_eigenpairs_estim(apply_A, A_shape, A_device, A_dtype, 
                          k = 1, cg_n_iter = 10, max_inv_iter = 100, 
                          eps = 1e-12, rayleigh_apply_A = None, 
                          sing_thres = 1e-2, 
                          verbose = False):
    """
    Recursively computing the minimial eigenpairs of the matrix A from apply_A

    Parameters
    ----------
    apply_A : function
        function of matrix multiplication by A.
    A_shape : tuple
        (M, N).
    k : int, optional
        Number of desired eigenpairs. The default is 1.
    cg_n_iter : int, optional
        number of conjugate gradient steps to estimate the inverse. The default is 10.
    max_inv_iter : int, optional
        maximal number of inverse power iterations. The default is 100.
    dtype : type, optional
        The default is torch.float64.
    device : device, optional
        The default is "cuda".
    eps : float, optional
        tolerance threshold. The default is 1e-12.
    rayleigh_apply_A : non-user variable, optional
        DESCRIPTION. The default is None.

    Return : List of eigenpairs sorted from the lowest to the highest
    -------
    TYPE
        DESCRIPTION.

    """
    
    y1, l_min1 = _min_eigenpair_estim(apply_A, A_shape, A_device, A_dtype, 
                                      cg_n_iter, max_inv_iter, eps = eps,
                                      sing_thres = sing_thres,
                                      verbose = verbose)
    if rayleigh_apply_A != None:
        l_min1 = matrix_free_rayleight(rayleigh_apply_A, y1).item()
        #(torch.linalg.norm(rayleigh_apply_A(y1)) / torch.linalg.norm(y1)).item()
    else:
        rayleigh_apply_A = apply_A
    
    if k == 1:
        return [(y1, l_min1)]
    else:
        reduced_apply_A = lambda x : apply_A(x) + 1e10 * y1[:, None] @ (y1[None, :] @ x) 
        eigen = _min_eigenpairs_estim(reduced_apply_A, A_shape, A_device, 
                                      A_dtype, k - 1, cg_n_iter, max_inv_iter, 
                                      eps, rayleigh_apply_A, sing_thres, verbose)
        return [(y1, l_min1)] + eigen


def _min_eigenpair_estim(apply_A, A_shape, A_device, A_dtype, cg_n_iter = 10, 
                         max_inv_iter = 100, eps = 1e-12, sing_thres = 1e-2,
                         verbose = False):
    M, N = A_shape
    assert M == N
    
    x = torch.randn((N,), dtype = A_dtype, device = A_device)
    x = x / x.norm()
    y = _conj_grad_estim(apply_A, x, x, eps, cg_n_iter, sing_thres, verbose)
    y = y / torch.linalg.norm(y)
    
    Ay = apply_A(y)
    l = torch.dot(Ay, y)/ torch.sum(y**2)
    stop_criteria = (torch.linalg.norm(Ay - l * y) <= l * eps)
    
    i = 0
    while not(stop_criteria) and i < max_inv_iter:
        y = _conj_grad_estim(apply_A, y, y, eps, cg_n_iter, sing_thres, verbose)
        y = y / torch.linalg.norm(y)
        i += 1
        
        Ay = apply_A(y)
        l = torch.dot(Ay, y)/ torch.sum(y**2)
        stop_criteria = (torch.linalg.norm(Ay - l * y) <= l * eps)
        
    l_min = matrix_free_rayleight(apply_A, y).item()
    return y, l_min


def _conj_grad_estim(apply_A, x, b, tol, N, sing_thres, show = False):
    r = b - apply_A(x)
    p = torch.clone(r)
    if show:
        T = tqdm(range(N))
    else:
        T = range(N)
    for i in T:
        Ap = apply_A(p)
        alpha = torch.dot(p, r) / torch.dot(p, Ap)
        x = x + alpha * p
        r = b - apply_A(x)
        
        rayleight = matrix_free_rayleight(apply_A, x).item()
        
        stop_crit = (torch.sqrt(torch.sum((r**2))) < tol
                     ) or (rayleight <= sing_thres)
        if stop_crit:
            break
        else:
            beta = -torch.dot(r, Ap) / torch.dot(p, Ap)
            p = r + beta * p
        
        if show:
            T.set_description(
                f"conj_grad, singval: {rayleight**0.5}")
        
    return x 
