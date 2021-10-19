import numpy as np
from nptyping import NDArray
import random
import numpy.linalg as LA
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from utils import load_image_to_arr, display_rgb, SVD2D
import os
import zipfile
from PIL import Image
from io import BytesIO ## for Python 3
import re

visual_data_zip_path: str = None
visual_data_zipfile: zipfile.ZipFile = None
def set_visual_data_zip_path(zippath: str):
    global visual_data_zip_path
    global visual_data_zipfile
    visual_data_zip_path = zippath
    visual_data_zipfile = zipfile.ZipFile(zippath)

semantic_data_zip_path: str = None
semantic_data_zipfile: zipfile.ZipFile = None
def set_semantic_data_zip_path(zippath: str):
    global semantic_data_zip_path
    global semantic_data_zipfile
    semantic_data_zip_path = zippath
    semantic_data_zipfile = zipfile.ZipFile(zippath)

get_sceneid_from_visual_filename_regex = r"(.+)\/(.+)_(\d+)_leftImg8bit.png"
get_sceneid_from_visual_filename = lambda fname: re.findall(get_sceneid_from_visual_filename_regex, fname)[0][1]
get_sceneid_from_semantic_filename_regex = r"(.+)\/(.+)_(\d+)_gtFine_labelIds.png"
get_sceneid_from_semantic_filename = lambda fname: re.findall(get_sceneid_from_semantic_filename_regex, fname)[0][1]

test_data_index: Dict[str, Dict[str, str]] = {}
train_data_index: Dict[str, Dict[str, str]] = {}
sceneIDs: Set[str] = set()
def initialize_tool():
    global test_data_index
    global train_data_index
    global sceneIDs
    test_data_index = {}
    train_data_index = {}
    sceneIDs = set()

    vis_file_names = [f.filename for f in visual_data_zipfile.filelist if "leftImg8bit.png" in f.filename]
    vis_file_directory = {get_sceneid_from_visual_filename(fname):fname for fname in vis_file_names}

    sem_file_names = [f.filename for f in semantic_data_zipfile.filelist if "labelIds.png" in f.filename]
    sem_file_directory = {get_sceneid_from_semantic_filename(fname):fname for fname in sem_file_names}

    for sceneID in vis_file_directory:
        sceneIDs.add(sceneID)
        is_train = "train" in vis_file_directory[sceneID]
        info = {"visualFile": vis_file_directory[sceneID], "semanticFile": sem_file_directory[sceneID]}
        if(is_train):
            train_data_index[sceneID] = info
        else:
            test_data_index[sceneID] = info


def loadVisualInfo(sceneID: str):
    isTest = sceneID in test_data_index
    path = test_data_index[sceneID]["visualFile"] if isTest else train_data_index[sceneID]["visualFile"]
    img_bytes = BytesIO(visual_data_zipfile.read(path))
    img = Image.open(img_bytes)
    img_arr = np.asarray(img)
    return img_arr

def loadSemanticInfo(sceneID: str):
    isTest = sceneID in test_data_index
    path = test_data_index[sceneID]["semanticFile"] if isTest else train_data_index[sceneID]["semanticFile"]
    img_bytes = BytesIO(semantic_data_zipfile.read(path))
    img = Image.open(img_bytes)
    img_arr = np.asarray(img)
    return img_arr

def loadScene(sceneID: str) -> Tuple[NDArray[Any], NDArray[Any]]:
    return (loadVisualInfo(sceneID), loadSemanticInfo(sceneID))

def loadRandomScene() -> Tuple[ str, Tuple[NDArray[Any], NDArray[Any]] ]:
    sceneID = random.sample(sceneIDs, 1)[0]
    return (sceneID, loadScene(sceneID))

def loadRandomVisualInfo() -> Tuple[str, NDArray[Any]]:
    sceneID = random.sample(sceneIDs, 1)[0]
    return (sceneID, loadVisualInfo(sceneID))

