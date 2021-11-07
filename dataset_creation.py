import numpy as np
from nptyping import NDArray
import numpy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image
import random
from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from utils import load_image_to_arr, display_rgb, SVD2D
import gc
import dim_reduction
from spectral_data import ImageSpectralData
dim_reduction.init_basis_cache()
import cityscapes_helper
import os
import json

CITYSCAPES_Y = 1024
CITYSCAPES_X = 2048

def numpy_upsample_2d(A, upsample_factor):
    A_up_x = np.repeat(A, upsample_factor).reshape((A.shape[0], A.shape[1] * upsample_factor))
    return np.repeat(A_up_x, upsample_factor, axis=0).reshape((A_up_x.shape[0]*upsample_factor, A_up_x.shape[1]))

class ImageAtScale:
    def __init__(self, imageChunk: NDArray[Any], anchor: Tuple[int, int]):
        self.imageChunk: NDArray[Any] = imageChunk
        self.imageSize: int = imageChunk.shape[0]
        self.anchor: Tuple[int, int] = anchor
        self.chunk_spectral_data: ImageSpectralData = None
        self.downsample_factor: int = 1

    def set_downsample_factor(self, factor: int):
        self.downsample_factor = factor

    def compute_spectral_data(self):
        if(self.chunk_spectral_data is None):
            imgCnk = self.imageChunk
            if(self.downsample_factor > 1):
                imCnk = Image.fromarray(self.imageChunk)
                downsampled_dim1D = int(self.imageSize // self.downsample_factor)
                imgCnk = np.asarray(imCnk.resize((downsampled_dim1D, downsampled_dim1D), Image.NEAREST))
            self.chunk_spectral_data = ImageSpectralData(imgCnk)
        return self.chunk_spectral_data

class WholeImage(ImageAtScale):
    def __init__(self, image: NDArray[Any]):
        super().__init__(image, (0, 0))
        
    def compute_spectral_data(self):
        if(self.chunk_spectral_data is None):
            imCnk = Image.fromarray(self.imageChunk)
            downsampled_h = int(self.imageChunk.shape[0] // self.downsample_factor)
            downsampled_w = int(self.imageChunk.shape[1] // self.downsample_factor)
            imgCnk = np.asarray(imCnk.resize((downsampled_w, downsampled_h), Image.NEAREST))
            self.chunk_spectral_data = ImageSpectralData(imgCnk)
        return self.chunk_spectral_data

    

class ImageChunkAttributeSummarizer:
    def __init__(self, chunks: List[ImageAtScale], attribute_size: int, is_metadata: bool, description: str = "no summary", id: str = "ID"):
        self.description: str = description
        self.chunks: List[ImageAtScale] = chunks
        self.n_chunks: int = len(chunks)
        self.attribute_size = attribute_size
        self.is_metadata = is_metadata
        self.summary: NDArray[Any] = np.zeros((self.n_chunks, self.attribute_size))
        self.feature_id: str = id

    def calculate_summary(self) -> NDArray[Any]:
        pass


class ImageChunkMeanSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale]):
        description = "Image Chunk RGB Mean"
        summary_id = "mean"
        super().__init__(chunks, 3, False, description, summary_id)

    def calculate_summary(self):
        for i in range(self.n_chunks):
            spec_data = self.chunks[i].compute_spectral_data()
            self.summary[i] = spec_data.mean_color

class ImageChunkPrincipleColorLengthSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale], principle_color_index: int):
        description = f"The length of principle color {principle_color_index}"
        summary_id = f"pc_{principle_color_index}_len"
        super().__init__(chunks, 1, True, description, summary_id)
        self.principle_color_index = principle_color_index

    def calculate_summary(self) -> NDArray[Any]:
        for i in range(self.n_chunks):
            spec_data = self.chunks[i].compute_spectral_data()
            self.summary[i, 0] = spec_data.pc_lengths[self.principle_color_index]

class ImageChunkPrincipleColorSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale], principle_color_index: int) -> None:
        description = f"Priciple color {principle_color_index}"
        summary_id = f"pc_{principle_color_index}"
        super().__init__(chunks, 3, True, description, summary_id)
        self.principle_color_index = principle_color_index

    def calculate_summary(self) -> NDArray[Any]:
        for i in range(self.n_chunks):
            spec_data = self.chunks[i].compute_spectral_data()
            self.summary[i, :] = spec_data.principle_colors[self.principle_color_index]

class ImageChunkPCACoeffSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale], pc_index: int, n_coeffs: int):
        desc = f"{n_coeffs} PCA coefficients of this image chunks's {pc_index}th principle color channel"
        summary_id = f"pcm_{pc_index}_PCA_coeffs"
        super().__init__(chunks, n_coeffs, True, desc, summary_id)
        self.pc_index: int = pc_index
        self.n_coeffs: int = n_coeffs

    def calculate_summary(self) -> NDArray[Any]:
        for i in range(self.n_chunks):
            spec_data = self.chunks[i].compute_spectral_data()
            B, coeff = spec_data.get_pcm_pca_coeffs(self.pc_index, self.n_coeffs)
            self.summary[i, :] = coeff

class ImageChunkDownsampleFactorSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale], downsample_factor: int):
        desc = f"Downsample this image chunk by {downsample_factor}"
        summary_id = "downsample_factor"
        super().__init__(chunks, 1, True, desc, summary_id)
        self.downsample_factor: int = downsample_factor

    def calculate_summary(self) -> NDArray[Any]:
        for i in range(self.n_chunks):
            self.summary[i, 0] = self.downsample_factor

class ImageChunkAnchorSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale]) -> None:
        desc = "Top-Left Coordinate of Chunk"
        summary_id = "anchor"
        super().__init__(chunks, 2, True, desc, summary_id)
    
    def calculate_summary(self):
        for i in range(self.n_chunks):
            self.summary[i, 0] = self.chunks[i].anchor[0]
            self.summary[i, 1] = self.chunks[i].anchor[1]

class ImageChunkCenterLocationSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale]) -> None:
        desc = "Center Coordinate of Chunk"
        summary_id = "center_loc"
        super().__init__(chunks, 2, False, desc, summary_id)
    
    def calculate_summary(self):
        center_offset = self.chunks[0].imageSize / 2
        for i in range(self.n_chunks):
            self.summary[i, 0] = self.chunks[i].anchor[0] + center_offset
            self.summary[i, 1] = self.chunks[i].anchor[1] + center_offset

class ImageChunkCenterLocationPolarSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale]):
        desc = "Center Coordinate of Chunk in Polar corrds, reletive to the center of the image."
        summary_id = "center_loc_polar"
        super().__init__(chunks, 2, False, description=desc, id=summary_id)

    def calculate_summary(self) -> NDArray[Any]:
        anchor_center_offset = self.chunks[0].imageSize / 2
        full_y_center = CITYSCAPES_Y / 2
        full_x_center = CITYSCAPES_X / 2
        for i in range(self.n_chunks):
            chunk_center_x = self.chunks[i].anchor[0] + anchor_center_offset
            chunk_center_y = self.chunks[i].anchor[1] + anchor_center_offset
            chunk_x_center_offset = full_x_center - chunk_center_x
            chunk_y_center_offset = full_y_center - chunk_center_y

            dist = (chunk_x_center_offset**2 + chunk_y_center_offset**2) ** 0.5
            if(chunk_x_center_offset == 0):
                radians = np.sign(chunk_y_center_offset) * 1.57
            else:
                radians = np.arctan(chunk_y_center_offset / chunk_x_center_offset)
            self.summary[i, 0] = dist
            self.summary[i, 1] = radians


class ImageChunkSizeSummarizer(ImageChunkAttributeSummarizer):
    def __init__(self, chunks: List[ImageAtScale]) -> None:
        desc = "2D size of chunk"
        summary_id = "size2D"
        super().__init__(chunks, 2, False, desc, summary_id)

    def calculate_summary(self):
        h = self.chunks[0].imageChunk.shape[0]
        w = self.chunks[0].imageChunk.shape[1]
        self.summary[:, 0] = h
        self.summary[:, 1] = w

class MultiScaleImageChunkSummarizer:
    def __init__(self):
        pass

class ImageChunkSummarizerOptions:
    def __init__(self) -> None:
        self.include_mean = True
        self.included_principle_colors = [True, True, True]
        self.include_center = True
        self.include_center_polar = True
        self.include_anchor = True
        self.n_pcm_coeffs = [100, 35, 15]
        self.downsample_factor = 1


class ImageChunkEncoder:
    def __init__(self, options: ImageChunkSummarizerOptions):
        self.options: ImageChunkSummarizerOptions = options
        self.summarizers: List[ImageChunkAttributeSummarizer] = []
        self.feature_masks: Dict[str, NDArray[Any]] = {}
        self.num_total_features: int = -1
        
    def setup_summaries(self, chunks: List[ImageAtScale]):
        options = self.options
        n_chunk_summary_features = 0
        summarizers: List[ImageChunkAttributeSummarizer] = []
        summarizers.append(ImageChunkSizeSummarizer(chunks))
        summarizers.append(ImageChunkDownsampleFactorSummarizer(chunks, options.downsample_factor))
        if(options.include_mean):
            summarizers.append(ImageChunkMeanSummarizer(chunks))
        for pci in range(3):
            if(options.included_principle_colors[pci]):
                summarizers.append(ImageChunkPrincipleColorLengthSummarizer(chunks, pci))
                summarizers.append(ImageChunkPrincipleColorSummarizer(chunks, pci))
                summarizers.append(ImageChunkPCACoeffSummarizer(chunks, pci, options.n_pcm_coeffs[pci]))
        if(options.include_center):
            summarizers.append(ImageChunkCenterLocationSummarizer(chunks))
        if(options.include_anchor):
            summarizers.append(ImageChunkAnchorSummarizer(chunks))
        if(options.include_center_polar):
            summarizers.append(ImageChunkCenterLocationPolarSummarizer(chunks))
        
        self.num_total_features = sum([s.attribute_size for s in summarizers])
        feature_index = 0
        for summarizer in summarizers:
            attr_size = summarizer.attribute_size
            self.feature_masks[summarizer.feature_id] = np.zeros(self.num_total_features, dtype=bool)
            self.feature_masks[summarizer.feature_id][feature_index:feature_index + attr_size] = True
            feature_index += attr_size

        self.summarizers = summarizers

    def encode_summaries(self, chunks: List[ImageAtScale]):
        result = np.zeros((len(chunks), self.num_total_features))
        for summarizer in self.summarizers:
            summarizer.calculate_summary()
            feature_mask = self.feature_masks[summarizer.feature_id]
            result[:, feature_mask] = summarizer.summary
        return result

class ImageChunkDecoder:
    def __init__(self, image_feature_masks: NDArray[Any]) -> None:
        self.feature_masks = image_feature_masks

    def decode_no_location(self, feature_vec: NDArray[Any]):
        h =  int(feature_vec[self.feature_masks["size2D"]][0])
        w =  int(feature_vec[self.feature_masks["size2D"]][1])
        is_square = h == w
        downsample_factor = int(feature_vec[self.feature_masks["downsample_factor"]])
        downsamele_h = int(h // downsample_factor)
        downsamele_w = int(w // downsample_factor)
        empty_spec_data = ImageSpectralData(None, (h, w))
        empty_spec_data.dim2D = (h, w)
        empty_spec_data.mean_color = feature_vec[self.feature_masks["mean"]]
        for pci in range(3):
            pcID = f"pc_{pci}"
            pclenID = f"pc_{pci}_len"
            pcCoeffID = f"pcm_{pci}_PCA_coeffs"
            if(pcID in self.feature_masks):
                empty_spec_data.principle_colors[pci] = feature_vec[self.feature_masks[pcID]]
            if(pclenID in self.feature_masks):
                empty_spec_data.pc_lengths[pci] = feature_vec[self.feature_masks[pclenID]]
            if(pcCoeffID in self.feature_masks):
                coeffs = feature_vec[self.feature_masks[pcCoeffID]]
                if(is_square):
                    B = dim_reduction.get_basis(downsamele_h)
                else:
                    B = dim_reduction.get_whole_image_basis(pci)
                pcm = B.get_approximation_from_coeffs(coeffs).reshape((downsamele_h, downsamele_w))
                empty_spec_data.PCMs[pci] = numpy_upsample_2d(pcm, downsample_factor)
        
        return empty_spec_data.get_realization()
        
class MultiScaleImageEncoder:
    def __init__(self, options: List[ImageChunkSummarizerOptions]):
        self.n_scales = len(options)
        self.options: List[ImageChunkSummarizerOptions] = options
        self.encoders: List[ImageChunkEncoder] = [ImageChunkEncoder(opt) for opt in options]
        self.scale_masks: List[NDArray[Any]] = []

    def encode(self, scale_chunks: List[List[ImageAtScale]]) -> NDArray[Any]:
        num_chunks = len(scale_chunks[0])
        for i in range(self.n_scales):
            chunks = scale_chunks[i]
            [c.set_downsample_factor(self.options[i].downsample_factor) for c in chunks]
            encoder = self.encoders[i]
            encoder.setup_summaries(chunks)
        
        total_feature_vec_size = sum([encoder.num_total_features for encoder in self.encoders])
        result = np.zeros((num_chunks, total_feature_vec_size))
        feature_index = 0
        for i in range(self.n_scales):
            encoder = self.encoders[i]
            chunks = scale_chunks[i]
            scale_mask = np.zeros(total_feature_vec_size, dtype=bool)
            num_scale_features = self.encoders[i].num_total_features
            scale_mask[feature_index:feature_index + num_scale_features] = True
            self.scale_masks.append(scale_mask)

            result[:, scale_mask] = encoder.encode_summaries(chunks)
            feature_index += num_scale_features
        return result


class MultiScaleImageDecoder:
    def __init__(self, scale_masks: List[NDArray[Any]], scale_feature_masks: List[Dict[str, NDArray[Any]]]):
        self.n_scales = len(scale_masks)
        self.scale_masks: List[NDArray] = scale_masks
        self.scale_feature_masks: List[Dict[str, NDArray[Any]]] = scale_feature_masks

    def decode(self, feature_vec: NDArray[Any]):
        result = np.zeros((CITYSCAPES_Y, CITYSCAPES_X, 3), dtype=np.uint8)
        for i in range(self.n_scales):
            single_scale_features = feature_vec[self.scale_masks[i]]
            single_scale_feature_mask = self.scale_feature_masks[i]
            image_chunk_anchor = single_scale_features[single_scale_feature_mask["anchor"]].astype(int)
            single_scale_decoder = ImageChunkDecoder(single_scale_feature_mask)
            single_scale_chunk = single_scale_decoder.decode_no_location(single_scale_features)
            h = single_scale_chunk.shape[0]
            w = single_scale_chunk.shape[1]
            result[image_chunk_anchor[1]:image_chunk_anchor[1]+h, image_chunk_anchor[0]:image_chunk_anchor[0]+w, :] = single_scale_chunk
        return result


class MultiScaleImageSampler:
    def __init__(self, scales: List[int]) -> None:
        #pre-condition: All scales are even numbers.
        self.scales: List[int] = scales
        self.num_scales: int = len(scales)
        self.first_scale = scales[0]

    def sample(self, image: NDArray[Any], anchor: Tuple[int, int] = None) -> List[ImageAtScale]:
        imgY = image.shape[0]
        imgX = image.shape[1]
        anchors: List[Tuple[int, int]] = []
        result: List[ImageAtScale] = []
        
        if(anchor is None):
            first_anchor_X = random.randint(0, imgX - self.first_scale)
            first_anchor_Y = random.randint(0, imgY - self.first_scale)
        else:
            first_anchor_X = anchor[1]
            first_anchor_Y = anchor[0]
        anchors.append((first_anchor_X, first_anchor_Y))

        for i in range(1, self.num_scales):
            prev_anchor_x, prev_anchor_y = anchors[i-1]
            prev_scale = self.scales[i-1]
            scale = self.scales[i]
            anchor_x = prev_anchor_x + ((prev_scale - scale) // 2)
            anchor_y = prev_anchor_y + ((prev_scale - scale) // 2)
            anchors.append((anchor_x, anchor_y))
    
        for i in range(self.num_scales):
            anchor_x, anchor_y = anchors[i]
            scale = self.scales[i]
            chunk = image[anchor_y:anchor_y+scale, anchor_x:anchor_x+scale, :]
            result.append(ImageAtScale(chunk, anchors[i]))
        
        return result
        
def semantic_image_composition(semantic_img_composition: NDArray[Any]) -> NDArray[Any]:
    v = semantic_img_composition.ravel()
    result = np.zeros(30)
    for i in range(30):
        result[i] = np.sum(v == i)
    result = result / len(v)
    return result

def numpy_mask_to_01_str(np_mask: NDArray[Any]) -> str:
    zero_one = list(np_mask.astype(int))
    zero_one_str = [str(x) for x in zero_one]
    return "".join(zero_one_str)

def zero_one_str_to_numpy_mask(zero_one_str: str) -> NDArray[Any]:
    zero_one = [int(s) for s in zero_one_str]
    return np.array(zero_one, dtype=bool)

class CityscapesDatasetFactory:
    def __init__(self, dataset_id: str, scales: List[int], summarizer_options: List[ImageChunkSummarizerOptions], whole_image_summarizer_options: ImageChunkSummarizerOptions = None):
        self.dataset_id: str = dataset_id
        self.scales: List[int] = scales
        self.n_scales: int = len(scales)
        self.coreDim1d: int = self.scales[-1]
        self.summarizer_options: List[ImageChunkSummarizerOptions] = summarizer_options
        self.sampler: MultiScaleImageSampler = MultiScaleImageSampler(scales)

        self.whole_image_summarizer_options = whole_image_summarizer_options
        if(self.whole_image_summarizer_options is not None):
            self.n_scales += 1
            self.summarizer_options.insert(0, whole_image_summarizer_options)

        self.feature_vects: NDArray[Any] = None
        self.scale_masks: List[NDArray[Any]] = None
        self.feature_masks: List[Dict[str, NDArray[Any]]] = None
        self.core_compositions: NDArray[Any] = None
        self.sceneIDs: List[str] = None

        self.batch_size = 50
        self.samples_from_image = 10

    def create_dataset(self, n_observations: int):
        self.sceneIDs = []
        n_obs_processed = 0
        n_batches = n_observations // self.batch_size
        while n_obs_processed < n_observations:
            batch_index = 0
            image_sample_index = 0
            #for every batch
            for batch_index in range(n_observations // self.batch_size):
                batch_core_compositions = []
                batch_feature_vects = []
                batch_multi_scale_image_samples: List[List[ImageAtScale]] = [] #n_scales x batch_size
                for i in range(self.n_scales):
                    batch_multi_scale_image_samples.append([])

                #for every image in the batch
                for images_in_batch_index in range(self.batch_size // self.samples_from_image):
                    # load in an image
                    (sceneID, (visualImg, semanticImg)) = cityscapes_helper.loadRandomScene()
                    whole_image_scale = WholeImage(visualImg)
                    for image_sample_index in range(self.samples_from_image):
                        #sample like 10 observations from that image
                        self.sceneIDs.append(sceneID)
                        visualScales = self.sampler.sample(visualImg)
                        if(self.whole_image_summarizer_options is not None):
                            visualScales.insert(0, whole_image_scale)
                        for si in range(self.n_scales):
                            batch_multi_scale_image_samples[si].append(visualScales[si])
                        (coreAnchorX, coreAnchorY) = visualScales[-1].anchor #anchor is x, y
                        semanticImg_core = semanticImg[coreAnchorY:coreAnchorY+self.coreDim1d, coreAnchorX:coreAnchorX+self.coreDim1d]
                        semanticComposition = semantic_image_composition(semanticImg_core)
                        batch_core_compositions.append(semanticComposition)
                        n_obs_processed += 1
                
                if(self.core_compositions is None):
                    self.core_compositions = np.vstack(batch_core_compositions)
                else:
                    self.core_compositions = np.vstack([self.core_compositions, batch_core_compositions])

                multi_chunk_encoder = MultiScaleImageEncoder(self.summarizer_options)
                batch_feature_vects = multi_chunk_encoder.encode(batch_multi_scale_image_samples)
                if(self.feature_vects is None):
                    self.feature_vects = np.vstack(batch_feature_vects)
                else:
                    self.feature_vects = np.vstack([self.feature_vects, batch_feature_vects])
                self.scale_masks = multi_chunk_encoder.scale_masks
                self.feature_masks = [enc.feature_masks for enc in multi_chunk_encoder.encoders]

                #free memory from the batch
                del batch_core_compositions
                del batch_feature_vects
                del batch_multi_scale_image_samples
                gc.collect()
                
                print(f"processed batch {batch_index + 1} of {n_batches}")


    def create_config_dict(self):
        result = {}
        result["name"] = self.dataset_id
        result["scale_masks"] = [numpy_mask_to_01_str(mask) for mask in self.scale_masks]
        result["feature_masks"] = []
        for si in range(self.n_scales):
            #{key:value for (key,value) in dictonary.items()}
            d = {attrID:numpy_mask_to_01_str(mask) for (attrID, mask) in self.feature_masks[si].items()}
            result["feature_masks"].append(d)
        result["sceneIDs"] = self.sceneIDs
        return result

    def split_feature_vects(self):
        result = []
        MAX_FILE_VECTS = 5000
        n_features = self.feature_vects.shape[0]
        n_splits = (n_features // MAX_FILE_VECTS) + 1
        for split_i in range(n_splits):
            feature_split = self.feature_vects[split_i*MAX_FILE_VECTS: min((split_i+1)*MAX_FILE_VECTS, n_features)]
            result.append(feature_split)
        return result

    def persist_dataset(self):
        folder_path = os.path.join("datasets", self.dataset_id)
        os.mkdir(folder_path)
        
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.create_config_dict(), f)

        features_folder_path = os.path.join(folder_path, "features")
        os.mkdir(features_folder_path)
        split_features = self.split_feature_vects()
        n_splits = len(split_features)
        for split_i in range(n_splits):
            feature_file_path = os.path.join(features_folder_path, f"{str(split_i)}.npy")
            with open(feature_file_path, "wb") as f:
                np.save(f, split_features[split_i].astype(np.float32))

        labels_path = os.path.join(folder_path, "labels.npy")
        with open(labels_path, "wb") as f:
            np.save(f, self.core_compositions)
        

class CityScapesDataset:
    def __init__(self, dataset_id: str):
        self.dataset_id: str = dataset_id

        self.features: NDArray[Any] = None
        self.labels: NDArray[Any] = None
        self.decoder: MultiScaleImageDecoder = None
        self.sceneIds: List[str] = None
        self.n_scales: int = None

        self.scale_masks: List[NDArray[Any]] = []
        self.feature_masks: List[Dict[str, NDArray[Any]]] = []

        self.raw_feature_min: NDArray[Any] = None 
        self.raw_feature_max: NDArray[Any] = None
        self.normalized_features: NDArray[Any] = None

    def load(self):
        folder_path = os.path.join("datasets", self.dataset_id)
        
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.sceneIds = config["sceneIDs"]
        self.n_scales = len(config["scale_masks"])
        scale_masks = []
        feature_masks = []
        for si in range(self.n_scales):
            scale_mask = zero_one_str_to_numpy_mask(config["scale_masks"][si])
            scale_masks.append(scale_mask)
            #{attrID:numpy_mask_to_01_str(mask) for (attrID, mask) in self.feature_masks[si].items()}
            feature_maskz = {attrID:zero_one_str_to_numpy_mask(mask_01_str) for (attrID, mask_01_str) in config["feature_masks"][si].items()}
            feature_masks.append(feature_maskz)
        self.decoder = MultiScaleImageDecoder(scale_masks, feature_masks)
        self.scale_masks = scale_masks
        self.feature_masks = feature_masks

        labels_path = os.path.join(folder_path, "labels.npy")
        with open(labels_path, "rb") as f:
            self.labels = np.load(f)
        
        feature_folder = os.path.join(folder_path, "features")
        feature_files = os.listdir(feature_folder)
        feature_file_number = lambda ff: int(ff.split(".")[0])
        feature_files = sorted(feature_files, key=feature_file_number)
        feature_splits = []
        for feature_file in feature_files:
            print(feature_file)
            feature_file_path = os.path.join(feature_folder, feature_file) 
            with open(feature_file_path, "rb") as f:
                feature_splits.append(np.load(f))
        self.features = np.vstack(feature_splits)
        self.compute_normalized_features()

    def get_decoded_feature(self, feature_i):
        return self.decoder.decode(self.features[feature_i])

    def get_feature_original_image(self, feature_i):
        sceneID = self.sceneIds[feature_i]
        return cityscapes_helper.loadVisualInfo(sceneID)

    def compute_normalized_features(self):
        features = self.features
        self.raw_feature_min = np.min(self.features, axis=0)
        self.raw_feature_max = np.max(self.features, axis=0)

        feature_range = self.raw_feature_max - self.raw_feature_min
        feature_01 = (features - self.raw_feature_min) / feature_range
        feature_neg1_to_1 = (feature_01 - 0.5) * 2
        self.normalized_features = feature_neg1_to_1

    def normalize_features(self, features: NDArray[Any]):
        dataset_feature_range = self.raw_feature_max - self.raw_feature_min
        feature_01 = (features - self.raw_feature_min) / dataset_feature_range
        feature_neg1_to_1 = (feature_01 - 0.5) * 2
        return feature_neg1_to_1



    def get_normalized_features(self):
        features = self.features
        feature_min = np.min(self.features, axis=0)
        feature_max = np.max(self.features, axis=0)
        feature_range = feature_max - feature_min
        feature_01 = (features - feature_min) / feature_range
        feature_neg1_to_1 = (feature_01 - 0.5) * 2
        return feature_neg1_to_1

    def get_feature_subset_mask(self, feature_subset: List[Tuple[int, str]]) -> NDArray[Any]:
        # each item in feature_subset is (scale index, featureID)
        feature_dim = self.features.shape[1]
        result = np.zeros(feature_dim, dtype = bool)
        for feature in feature_subset:
            (scale_index, feature_id) = feature
            scale_mask_start = self.scale_masks[scale_index].argmax()
            scale_feature_mask_start = self.feature_masks[scale_index][feature_id].argmax()
            scale_feature_len = np.sum(self.feature_masks[scale_index][feature_id])
            full_mask_start = scale_mask_start + scale_feature_mask_start
            result[full_mask_start:full_mask_start+scale_feature_len] = True
        return result
            


