import torch
import pytest
from time import time

from ..src.jackpot.torch_lobpcg import lobpcg, lobpcg_estim, matrix_free_rayleight


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
