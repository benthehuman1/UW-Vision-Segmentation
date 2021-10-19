import imp
import numpy as np
import scipy

def dct2D(a):
    return scipy.fft.dct( scipy.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2D(a):
    return scipy.fft.idct( scipy.fft.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

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

def truncDCT2D(A):
    all_coeffs = scipy.fft.dct( scipy.fft.dct(A, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    snake = getDCTSnake(A.shape[0])
    



