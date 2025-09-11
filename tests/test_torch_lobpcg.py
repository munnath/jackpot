import torch
import pytest
from time import time

from ..src.jackpot.torch_lobpcg import lobpcg, lobpcg_estim, matrix_free_rayleight


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

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lobpcg_matches_svd(dtype):
    """Check that custom lobpcg approximates SVD eigenvalues."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M, N = 500, 500
    k = 3

    # Construct symmetric PSD matrix
    A_sub = torch.randn((M, N), dtype=dtype, device=device)
    A = A_sub.T @ A_sub

    apply_A = lambda x: A @ x
    A_shape = A.shape

    # Custom lobpcg
    with torch.no_grad():
        eigvals_custom, _ = lobpcg(
            apply_A, A_shape, device, dtype, k=k, largest=False, method="ortho"
        )

    # Torch reference
    with torch.no_grad():
        eigvals_torch, _ = torch.lobpcg(
            A, k=k, largest=False, method="ortho"
        )

    # SVD reference (smallest k)
    with torch.no_grad():
        eig_svd = (
            torch.linalg.svd(A, full_matrices=False).S[-k:].to(dtype=dtype, device=device)
        )

    # Verify order & relative error tolerance
    eigvals_custom = eigvals_custom.cpu()
    eigvals_torch = eigvals_torch.cpu()
    eig_svd = eig_svd.cpu()

    assert torch.allclose(
        eigvals_custom.sort().values, eig_svd.sort().values, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(
        eigvals_torch.sort().values, eig_svd.sort().values, rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    dtype = torch.float64
    M, N = 2000, 2000
    k = 3

    A_sub = torch.randn((M, N), dtype=dtype, device=device)
    A = A_sub.T @ A_sub

    apply_A = lambda x: A @ x
    A_shape = A.shape

    # Custom lobpcg
    t1 = time()
    with torch.no_grad():
        a = lobpcg(apply_A, A_shape, device, dtype, k=k, largest=False, method="ortho")
    t_lob = time() - t1

    # Estimation
    t1 = time()
    with torch.no_grad():
        b = lobpcg_estim(apply_A, A_shape, device, dtype, k=k, cg_n_iter=1000, max_inv_iter=10)
    t_est = time() - t1

    # Torch lobpcg
    t1 = time()
    with torch.no_grad():
        c = torch.lobpcg(A, k=k, largest=False, method="ortho")
    t_torch = time() - t1

    # SVD baseline
    t1 = time()
    with torch.no_grad():
        eig = torch.linalg.svd(A, full_matrices=False).S[-k:].tolist()[::-1]
    t_svd = time() - t1

    print(f"lobpcg: {a[0].tolist()} \ntime: {t_lob:.3f}s")
    print(f"torch lobpcg: {c[0].tolist()} \ntime: {t_torch:.3f}s")
    print(f"estimation: {b[0].tolist()} \ntime: {t_est:.3f}s")
    print(f"svd: {eig} \ntime: {t_svd:.3f}s\n")

    print("Rayleigh norms verification:")
    print(f"lobpcg: {matrix_free_rayleight(apply_A, a[1]).tolist()}")
    print(f"torch lobpcg: {matrix_free_rayleight(apply_A, c[1]).tolist()}")
    print(f"estimation: {matrix_free_rayleight(apply_A, b[1]).tolist()}")
    print(f"svd: {eig}")
