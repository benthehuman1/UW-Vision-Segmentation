from turtle import color
import numpy as np
from nptyping import NDArray
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from utils import load_image_to_arr, display_rgb, SVD2D
import dim_reduction

class ImageSpectralData:
    def __init__(self, img: NDArray[Any]):
        self.dim1D = img.shape[0]
        self.mean_color: NDArray[Any] = None
        self.principle_colors: NDArray[Any] = None #3x3, unit length
        self.pc_lengths: NDArray[Any] = None
        self.PCMs: NDArray[Any] = None #3xdimxdim
        self.pc_lengths_standardized: NDArray[Any] = None

        img_area_2d = img.shape[0] * img.shape[1]
        colors = img.reshape(img_area_2d, 3)
        self.mean_color = np.mean(colors, axis=0)
        colors_mean_rem = colors - self.mean_color
        color_svd = SVD2D(colors_mean_rem)

        self.principle_colors = color_svd.VT
        self.pc_lengths = color_svd.s
        self.PCMs = np.zeros((3, self.dim1D, self.dim1D))
        for i in range(3):
            self.PCMs[i] = color_svd.U[:, i].reshape((self.dim1D, self.dim1D))

        self.pc_lengths_standardized = ((self.pc_lengths ** 2) / (img_area_2d * 3)) ** 0.5

    def get_pcm_pca_coeffs(self, pcm_index: int, num_coeffs: int):
        dim_reduction.init_basis_cache()
        basis = dim_reduction.get_basis(self.dim1D)
        return basis, basis.get_basis_coeffs(self.PCMs[pcm_index].ravel(), num_coeffs)
        