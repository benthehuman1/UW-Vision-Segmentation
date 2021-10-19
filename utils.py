from PIL import Image, ImageOps
import numpy as np
import numpy.linalg as LA
from nptyping import NDArray
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def load_image_to_arr(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image)

def display_rgb(A, ax=None):
    if(ax==None):
        fig, ax = plt.subplots()
    ax.imshow(A, norm=colors.Normalize(vmin=0, vmax=255))


def dct_snake(shape1D):
    result = np.zeros((shape1D, shape1D), dtype=int) - 1
    isUpSnake = True
    currX = 0
    currY = 0
    i = 0
    while(i < ((shape1D**2)/2)-2 ):
        result[currX, currY] = i
        isOnTopEdge = currY == 0
        isOnLeftEdge = currX == 0
        if(isUpSnake):
            if(isOnTopEdge):
                currX += 1
                isUpSnake = False
            else:
                currX += 1
                currY -= 1
        else:
            if(isOnLeftEdge):
                currY += 1
                isUpSnake = True
            else:
                currX -= 1
                currY += 1
        i += 1
    return result

def getDCTSnake(dim):
    with open(f"dct2DSnakeCache/dim{dim}.npy") as f:
        return np.load(f)



class SVD2D:
    def __init__(self, A: NDArray[Any, Any]) -> None:
        self.shape2d = A.shape
        self.U, self.s, self.VT = LA.svd(A, full_matrices=False)
        
    def low_rank(self, rank = "F"):
        if(rank == "F"):
            return (self.s * self.U).dot(self.VT)
        elif rank == 0:
            return np.zeros(self.shape2d)
        else:
            U_trunc = self.U[:, 0:rank]
            s_trunc = self.s[0:rank]
            VT_trunc = self.VT[0:rank, :]
            return (s_trunc * U_trunc).dot(VT_trunc)