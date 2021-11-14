import numpy as np
from nptyping import NDArray
import random
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from PIL import Image
import os
import json
from utils import load_image_to_arr, display_rgb, SVD2D
import dim_reduction
from spectral_data import ImageSpectralData
dim_reduction.init_basis_cache()
import cityscapes_helper
from dataset_creation import ( 
    MultiScaleImageEncoder,
    MultiScaleMultiResolutionImageSampler,
    DownsampledImageAtScale,
    ImageAtScale, 
    WholeImage,
    CityScapesDataset,
    semantic_image_composition
)
from cityscapes_helper import format_semantic_composition


def anchor_grid(largest_scale: int, smallest_scale: int) -> List[List[Tuple[int, int]]]:
    CITYSCAPES_Y = 1024
    CITYSCAPES_X = 2048

    #total_margin = largest_scale - smallest_scale
    num_rows = (CITYSCAPES_Y - largest_scale) // smallest_scale
    num_cols = (CITYSCAPES_X - largest_scale) // smallest_scale
    result = []
    for row_i in range(num_rows):
        curr_row = []
        anchor_y = row_i * smallest_scale
        for col_i in range(num_cols):
            anchor_x = col_i * smallest_scale
            curr_row.append([anchor_y, anchor_x])
        result.append(curr_row)
    return result

def multiscale_image_grid(anchor_grid: List[List[Tuple[int, int]]], sampler: MultiScaleMultiResolutionImageSampler, img: NDArray[Any]) -> List[List[List[DownsampledImageAtScale]]]:
    #flat_anchors =  [item for sublist in anchor_grid for item in sublist]
    #flat_multiscale_images = [sampler.sample(img, anchor) for anchor in flat_anchors]
    i = 0
    result = []
    whole_image_scale = WholeImage(img)
    for anchor_row in anchor_grid:
        image_row = []
        for anchor in anchor_row:
            multiscaleImages = sampler.sample(img, anchor)
            multiscaleImages.insert(0, whole_image_scale)
            image_row.append(multiscaleImages)
        result.append(image_row)
    return result

def semantic_composition_grid(sceneID: str, scales: List[int]):
    smallest_scale = scales[-1]
    core_oneside_margin = int((scales[0] - smallest_scale) / 2)
    semantic_composition = cityscapes_helper.loadSemanticInfo(sceneID)
    ag = anchor_grid(scales[0], scales[-1])
    nrows = len(ag)
    ncols = len(ag[0])
    num_classes = 30
    result = np.zeros((nrows, ncols, num_classes))
    for row_i in range(nrows):
        for col_i in range(ncols):
            anchor_Y, anchor_X = ag[row_i][col_i]
            core_anchor_Y = anchor_Y + core_oneside_margin
            core_anchor_X = anchor_X + core_oneside_margin
            semantic_core = semantic_composition[core_anchor_Y:core_anchor_Y+smallest_scale, core_anchor_X:core_anchor_X+smallest_scale]
            result[row_i, col_i, :] = semantic_image_composition(semantic_core)
    return result


def feature_grid(sceneID: str, encoder: MultiScaleImageEncoder, scales: List[int], downsample_factors: List[int]):
    image = cityscapes_helper.loadVisualInfo(sceneID)
    sampler = MultiScaleMultiResolutionImageSampler(scales, downsample_factors, image)
    sampler.compute_downsamples()
    ag = anchor_grid(scales[0], scales[-1])
    image_grid = multiscale_image_grid(ag, sampler, image)
    nrows = len(image_grid)
    ncols = len(image_grid[0])

    dumbo = [[scaleImage] for scaleImage in image_grid[0][0]]
    first_feature = encoder.encode(dumbo)
    feature_size = first_feature.shape[1]

    wumb = 0
    n_all_scales = len(image_grid[0][0])
    flattened_chunks: List[List[ImageAtScale]] = [[] for i in range(len(image_grid[0][0]))]
    for row_i in range(nrows):
        for col_i in range(ncols):
            for scale_i in range(n_all_scales):
                lil = image_grid[row_i][col_i][scale_i]
                flattened_chunks[scale_i].append(lil)
    encoded_features_tall = encoder.encode(flattened_chunks)
    wumb = 0
    result = np.zeros((len(image_grid), len(image_grid[0]), feature_size))
    for row_i in range(nrows):
        for col_i in range(ncols):
            result[row_i, col_i, :] = encoded_features_tall[wumb] 
            wumb += 1

    return result

def predict_semantic_grid(feature_grid: NDArray[Any], dataset: CityScapesDataset, use_normalized = True, prediction_fn = None):
    nrows = feature_grid.shape[0]
    ncols = feature_grid.shape[1]
    result = np.zeros((nrows, ncols, 30))
    r = dataset.raw_feature_max - dataset.raw_feature_min
    normalized_feature_grid = np.zeros(feature_grid.shape)
    
    for r in range(nrows):
        for c in range(ncols):
            feature = feature_grid[r, c, :]
            normalized_feature = dataset.normalize_features(np.array([feature]))
            normalized_feature_grid[r, c, :] = normalized_feature
     
    for r in range(nrows):
        for c in range(ncols):
            raw_feature = feature_grid[r, c, :]
            normalized_feature = normalized_feature_grid[r, c, :]

            if(use_normalized):
                prediction = prediction_fn(normalized_feature)
            else:
                prediction = prediction_fn(raw_feature)
            result[r, c, :] = prediction
    return result
