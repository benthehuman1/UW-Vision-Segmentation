from attr import has
import numpy as np
from nptyping import NDArray
from typing import List, Set, Dict, Tuple, Optional, Any, Callable

supported_basis_dims = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208]

class PartialOrthoBasis:
    def __init__(self, basis_vectors: NDArray[Any]):
        self.dimension = basis_vectors.shape[1]
        self.max_basis_size = basis_vectors.shape[0]
        self.basis_vectors = basis_vectors
    
    def get_basis_coeffs(self, observations: NDArray[Any], n_coeffs) -> NDArray[Any]:
        return np.dot(observations, self.basis_vectors[:n_coeffs].T)

    def get_approximations_from_coeffs(self, coeffs: NDArray[Any]):
        return np.dot(coeffs, self.basis_vectors)

    def get_approximation_from_coeffs(self, coeffs: NDArray[Any]):
        n_coeffs = coeffs.shape[0]
        return np.dot(coeffs, self.basis_vectors[:n_coeffs])

basis_cache: Dict[int, PartialOrthoBasis] = {}
has_initialized_cache = False
def init_basis_cache():
    global has_initialized_cache
    global supported_basis_dims
    if(has_initialized_cache):
        return
    for bd in supported_basis_dims:
        with open(f"pcm_bases/dim_{bd}.npy", "rb") as f:
            B = np.load(f)
            basis_cache[bd] = PartialOrthoBasis(B)

def get_basis(basisDim1D) -> PartialOrthoBasis:
    if(basisDim1D not in basis_cache):
        print("REEE INvalid basis")
        return None
    return basis_cache[basisDim1D]

