from cgitb import reset
from turtle import color
import numpy as np
from nptyping import NDArray
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from utils import load_image_to_arr, display_rgb, SVD2D
import dim_reduction

def trunc_color_rgb(A: NDArray[Any]):
    color = np.ravel(A)
    color[color < 0] = 0
    color[color > 255] = 255
    return color.reshape(A.shape)

class ImageSpectralData:
    def __init__(self, img: NDArray[Any] = None, dim1D: int = None):
        if(dim1D is None):
            dim1D = img.shape[0]
        self.dim1D: int = dim1D
        self.mean_color: NDArray[Any] = np.zeros(3)
        self.principle_colors: NDArray[Any] = np.zeros((3, 3)) #3x3, unit length
        self.pc_lengths: NDArray[Any] = np.zeros(3)
        self.PCMs: NDArray[Any] = np.zeros((3, self.dim1D, self.dim1D)) #3xdimxdim
        self.pc_lengths_standardized: NDArray[Any] = np.zeros(3)
        if(img is None):
            return

        self.dim1D = img.shape[0]
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

    def get_realization(self):
        shape3d = (self.dim1D, self.dim1D, 3)
        result = np.zeros(shape3d)
        for i in range(3):
            principle_color = self.principle_colors[i]
            pc_len = self.pc_lengths[i]
            result += (np.outer(self.PCMs[i], principle_color) * pc_len).reshape(shape3d)
        return trunc_color_rgb(result + self.mean_color).astype(np.uint8)
        
        

    """
    result = np.zeros(self.shape3d)
    for i in range(3):
      color_component = self.color_components[i]
      component_size = self.color_component_lengths[i]
      result += (np.outer(pcms[i].reshape(self.shape2d), color_component) * component_size).reshape(self.shape3d)
    return trunc_color_rgb(result + self.mean)
    """