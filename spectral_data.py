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
    def __init__(self, img: NDArray[Any] = None, dim2D: Tuple[int, int] = None):
        if(dim2D is None):
            dim2D = (img.shape[0], img.shape[1])
        self.dim2D: int = dim2D
        self.is_square = dim2D[0] == dim2D[1]
        self.mean_color: NDArray[Any] = np.zeros(3)
        self.principle_colors: NDArray[Any] = np.zeros((3, 3)) #3x3, unit length
        self.pc_lengths: NDArray[Any] = np.zeros(3)
        self.PCMs: NDArray[Any] = np.zeros((3, dim2D[0], dim2D[1])) #3xdimxdim
        self.pc_lengths_standardized: NDArray[Any] = np.zeros(3)
        if(img is None):
            return

        img_area_2d = dim2D[0] * dim2D[1]
        colors = img.reshape(img_area_2d, 3)
        self.mean_color = np.mean(colors, axis=0)
        colors_mean_rem = colors - self.mean_color
        color_svd = SVD2D(colors_mean_rem)

        self.principle_colors = color_svd.VT
        self.pc_lengths = color_svd.s
        self.PCMs = np.zeros((3, dim2D[0], dim2D[1]))
        for i in range(3):
            self.PCMs[i] = color_svd.U[:, i].reshape(dim2D)
            if(np.median(self.PCMs[i]) >= 0):
                self.PCMs[i] = -self.PCMs[i]
                self.principle_colors[i] = -self.principle_colors[i]

        self.pc_lengths_standardized = ((self.pc_lengths ** 2) / (img_area_2d * 3)) ** 0.5

    def get_pcm_pca_coeffs(self, pcm_index: int, num_coeffs: int):
        dim_reduction.init_basis_cache()
        if(self.is_square):
            basis = dim_reduction.get_basis(self.dim2D[0])
        else:
            #also pre-condition: is of shape 128x256 
            #different bases for pcm1, pcm2 etc. Terrible assumtion for resl-life, and bad code whatever IDK.
            basis = dim_reduction.get_whole_image_basis(pcm_index)
            pass
        return basis, basis.get_basis_coeffs(self.PCMs[pcm_index].ravel(), num_coeffs)

    def get_realization(self):
        shape3d = (self.dim2D[0], self.dim2D[1], 3)
        result = np.zeros(shape3d)
        for i in range(3):
            principle_color = self.principle_colors[i]
            pc_len = self.pc_lengths[i]
            result += (np.outer(self.PCMs[i], principle_color) * pc_len).reshape(shape3d)
        return trunc_color_rgb(result + self.mean_color).astype(np.uint8)
        
