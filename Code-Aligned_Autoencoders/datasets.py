import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
from itertools import count

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from change_priors import eval_prior, remove_borders, image_in_patches


def load_prior(name, expected_hw=None):
    """ Load prior from disk, validate dimensions and force shape to (h, w, 1) """
    mat = loadmat("data/" + name + "/change-prior.mat")
    varname = (
        re.sub(r"\W+", "", "aff" + str(expected_hw))
        if expected_hw is not None
        else "aff"
    )
    prior = tf.convert_to_tensor(mat[varname], dtype=tf.float32)
    if expected_hw is not None and prior.shape[:2] != expected_hw:
        raise FileNotFoundError
    if prior.ndim == 2:
        prior = prior[..., np.newaxis]
    return prior


def evaluate_prior(name, x, y, **kwargs):
    alpha = eval_prior(name, x, y, **kwargs)
    varname = re.sub(r"\W+", "", "aff" + str(x.shape[:2]))
    prior_path = "data/" + name + "/change-prior.mat"
    try:
        mat = loadmat(prior_path)
        mat.update({varname: alpha})
    except FileNotFoundError as e:
        mat = {varname: alpha}
    savemat(prior_path, mat)
    return alpha


def _denmark(reduce=False):
    """ Load Denmark dataset from .mat """
    mat = loadmat("data/Denmark/EMISAR_Foulum_PolSAR_logIntensity_CLband.mat")

    t1 = np.array(mat["imgCx"], dtype=np.single)
    t2 = np.array(mat["imgLy"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["GT"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _uk(reduce=True):
    """ Load UK dataset from .mat """
    mat = loadmat("data/UK/UK.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2[..., np.newaxis])
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (5, 5)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _italy(reduce=False):
    """ Load Italy dataset from .mat """
    mat = loadmat("data/Italy/Italy.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    if t1.shape[-1] == 3:
        t1 = t1[..., 0]
    t1, t2 = _clip(t1[..., np.newaxis]), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _france(reduce=True):
    """ Load France dataset from .mat """
    mat = loadmat("data/France/France.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (5, 5)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _california(reduce=False):
    """ Load California dataset from .mat """
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (4, 4)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _texas(clip=True):
    """ Load Texas dataset from .mat """
    mat = loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")

    t1 = np.array(mat["t1_L5"], dtype=np.single)
    t2 = np.array(mat["t2_ALI"], dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI_1"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask


def _polmak_ls5_s2_snap_collocate(reduce=False):
    """ Load California dataset from .mat """
    #import libtiff
    #from libtiff import TIFF
    from skimage import data, io, filters
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    
    print("3 December: Test normalised Polmak data with dummy change mask.")
    # Calefornia data
    #t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    #t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    #change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    #print(tf.reduce_max(t1),  tf.reduce_min(t1))
    #print(tf.reduce_max(t2),  tf.reduce_min(t2))
    #t1 = _clip(t1)
    #t2 = _clip(t2)
    #print(t1.shape)
    #print(t2.shape)
    #print(change_mask.shape)

    # Polmak data
    im = io.imread("data/Polmak/collocate_260717S2_030705LS5_v3.tif")
    changemap = io.imread("data/Polmak/ls5-s2-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    
    t1 = np.array(im[:, :, 10:17]) # np.array(im[:,:,10:17]) - correct size
    t2 = np.array(im[:, :, 0:10])

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    print(np.min(t1), np.max(t1))
    print(np.min(t2), np.max(t2))
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)
    #change_mask = np.eye(t1.shape[0], t1.shape[1])
    #change_mask = tf.convert_to_tensor(change_mask, dtype=tf.bool)
    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)
    #change_mask = tf.convert_to_tensor(np.ones(mat["ROI"].shape), dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")
        #reduction_ratios = (4, 4)
        #new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        #t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        #t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        #change_mask = tf.cast(
        #    tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
        #    tf.bool,
        #)

    return t1, t2, change_mask


def _polmak_ls5_s2_warp_align(reduce=False):
    """ Test LS5-S2 QGIS warp-aligned data. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/ls5-s2-qgis-warp-align.tif")
    changemap = io.imread("data/Polmak/ls5-s2-warp-align-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("16 December: Test LS5-S2 QGIS warp-aligned data.")
    
    t1 = np.array(im[:, :, 10:17]) # np.array(im[:,:,10:17]) - correct size
    t2 = np.array(im[:, :, 0:10])

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    print(np.min(t1), np.max(t1))
    print(np.min(t2), np.max(t2))
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_s2(reduce=False):
    """ Test LS5-S2 QGIS aligned data. """
    from skimage import data, io, filters


    # Polmak data
    im = io.imread("data/Polmak/ls5-s2-qgis-align.tif")
    changemap = io.imread("data/Polmak/ls5-s2-align-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("13 January: Test CLIP normalised LS5-S2 QGIS aligned data, no warp.")
    
    #t1 = np.array(im[:, :, 10:17]) 
    #t2 = np.array(im[:, :, 0:10])
    t1 = np.array(im[:, :-2, 10:17]) # np.array(im[:,:,10:17]) - correct size
    t2 = np.array(im[:, 2:, 0:10])
    changemap = changemap[:,1:-1,:]

    print("Bands t1 - before normalisation")
    _debug_print_bands(t1)
    print("Bands t2 - before normalisation")
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    #t1 = _norm01(t1, norm_type='band')
    #t1 = 2*t1 -1
    #t2 = _norm01(t2, norm_type='band')
    #t2 = 2*t2 -1

    print("Bands t1 - after normalisation")
    _debug_print_bands(t1)
    print("Bands t2 - after normalisation")
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)

    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_s2_ndvi(reduce=False):
    """ Test LS5-S2 QGIS aligned data. """
    from skimage import data, io, filters


    # Polmak data
    im = io.imread("data/Polmak/ls5-s2-qgis-align.tif")
    changemap = io.imread("data/Polmak/ls5-s2-align-changemap.tif")
    print("6 January: Test LS5-S2 NDVI (QGIS aligned data, no warp).")
    

    t1 = np.array(im[:, :-2, 10:17]) #t1 = np.array(im[:, :, 10:17]) 
    t2 = np.array(im[:, 2:, 0:10]) #t2 = np.array(im[:, :, 0:10])
    changemap = changemap[:,1:-1,:]

    # Calculate NDVI
    t1 = (t1[:,:, 3]-t1[:,:, 0])/(t1[:,:, 3]+t1[:,:, 0])
    t2 = (t2[:,:, 6]-t2[:,:, 2])/(t2[:,:, 6]+t2[:,:, 2])
    print(np.min(t1), np.mean(t1), np.max(t1))
    print(np.min(t2), np.mean(t2), np.max(t2))
    print(t1.shape)
    print(t2.shape)
    t1 = t1[..., np.newaxis]
    t2 = t2[..., np.newaxis]

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    print(np.min(t1), np.max(t1))
    print(np.min(t2), np.max(t2))
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_a2_s2_snap_collocate(reduce=False):
    """ Load Polmak AVNIR-2 - Sentinel-2 dataset. SNAP collocate."""
    from skimage import data, io, filters
    
    print("Test AVNIR-2 - S2 Polmak SNAP collocate data.")

    # Polmak data
    im = io.imread("data/Polmak/collocate_avnir2_s2_bandreduce.tif")
    changemap = io.imread("data/Polmak/a2-s2-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape)
    
    t1 = np.array(im[:,:,12:16]) 
    t2 = np.array(im[:,:,0:10])

    print(np.min(changemap[:,:,0]))
    print(np.min(changemap[:,:,1]))
    print(np.min(changemap[:,:,2]))
    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    print(np.min(t1), np.max(t1))
    print(np.min(t2), np.max(t2))
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask



def _polmak_a2_s2(reduce=False):
    """ Load Polmak AVNIR-2 - Sentinel-2 dataset. """
    from skimage import data, io, filters
    
    print("7 December: Test a2-s2-qgis-warp-align.")

    # Polmak data
    im = io.imread("data/Polmak/a2-s2-qgis-warp-align.tif")
    changemap = io.imread("data/Polmak/a2-s2-warp-align-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    
    t1 = np.array(im[:,:,0:4]) 
    t2 = np.array(im[:,:,6:16])

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    print(np.min(t1), np.max(t1))
    print(np.min(t2), np.max(t2))
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_pgnlma(reduce=False):
    """ Test LS5-S2 QGIS aligned data. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/collocate_LS5_PGNLM_A_19-2-64.tif")
    changemap = io.imread("data/Polmak/ls5-pgnlmA-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("7 January: Test LS5-PGNLM-A, log-transform collocate.")
    
    t1 = np.array(im[:, :, 11:18]) 
    t2 = np.array(im[:, :, 0:5])

    # Take loagrithm of intensity data (or is it amplitude input?)
    t2[:,:,0:4] = np.log(t2[:,:,0:4])

    # Debug, for printing values
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    # Debug, for printing values
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_pgnlmc(reduce=False):
    """ Test LS5-S2 QGIS aligned data. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/collocate_LS5_PGNLM_C_19-2-64.tif")
    changemap = io.imread("data/Polmak/ls5-pgnlmC-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("6 January: Test LS5-PGNLM-C, log-transform collocate.")
    
    t1 = np.array(im[:, :, 11:18]) 
    t2 = np.array(im[:, :, 0:5])

    # Take loagrithm of intensity data (or is it amplitude input?)
    t2[:,:,0:4] = np.log(t2[:,:,0:4])

    # Debug, for printing values
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    # Debug, for printing values
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_pgnlmc_stacked(reduce=False):
    """ Test LS5-S2 QGIS aligned data. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/collocate_LS5_PGNLM_C_19-2-64.tif")
    changemap = io.imread("data/Polmak/ls5-pgnlmC-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("12 January: Test LS5-PGNLM-C+S2, stacked, SAR log-transform collocate.")
    
    t1 = np.array(im[:, :, 11:18]) 
    t2 = np.array(im[:, :, 0:9])

    # Take loagrithm of intensity data (or is it amplitude input?)
    t2[:,:,0:4] = np.log(t2[:,:,0:4])

    # Debug, for printing values
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    # Debug, for printing values
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_ls5_pgnlma_stacked(reduce=False):
    """ Test LS5-S2 QGIS aligned data, stacked SAR and OPT. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/collocate_LS5_PGNLM_A_19-2-64.tif")
    changemap = io.imread("data/Polmak/ls5-pgnlmA-collocate-changemap.tif")
    print(im.shape)
    print(changemap.shape) # chans = change map, lat, long
    print("11 January: Test LS5-PGNLM-A+S2, stacked, SAR log-transform collocate.")
    
    t1 = np.array(im[:, :, 11:18]) 
    t2 = np.array(im[:, :, 0:9])

    # Take loagrithm of intensity data (or is it amplitude input?)
    t2[:,:,0:4] = np.log(t2[:,:,0:4])

    # Debug, for printing values
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = _norm01(t1, norm_type='band')
    t1 = 2*t1 -1
    t2 = _norm01(t2, norm_type='band')
    t2 = 2*t2 -1

    # Debug, for printing values
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(tf.reduce_max(t1),  tf.reduce_min(t1))
    print(tf.reduce_max(t2),  tf.reduce_min(t2))
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _polmak_pal_rs2_010817(reduce=False):
    """ Test PalSAR-RS2 data. """
    from skimage import data, io, filters

    # Polmak data
    im = io.imread("data/Polmak/subset_0_of_collocate_PalSAR_RS2-20170801.tif")
    changemap = io.imread("data/Polmak/pal-RS2_010817-collocate-changemap.tif")
    print("13 January: Test PalSAR-RS2 data, CLIP, channel-wise PalSAR negative value removal.")
    
    t1 = np.array(im[:, :, 7:9]) 
    t2 = np.array(im[:, :, 0:4])

    print('t1 - input values - PalSAR bands')
    _debug_print_bands(t1)
    print('t2 - input values - RADARSAT-2 bands')
    _debug_print_bands(t2)

    # Take loagrithm of intensity data 
    # To fix negative intensity values in PalSAR
    new_min = np.min(t2)
    t1[:,:,0] = t1[:,:,0] - np.min(t1[:,:,0]) + new_min
    t1[:,:,1] = t1[:,:,1] - np.min(t1[:,:,1]) + new_min
    t1= np.log(t1)
    t2= np.log(t2)

    # Debug, for printing values
    print('t1 - log - PalSAR bands')
    _debug_print_bands(t1)
    print('t2 - log - RADARSAT-2 bands')
    _debug_print_bands(t2)

    # Normalise to -1 to 1 range (for channels)
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    #t1 = _norm01(t1, norm_type='band')
    #t1 = 2*t1 -1
    #t2 = _norm01(t2, norm_type='band')
    #t2 = 2*t2 -1

    # Debug, for printing values
    print('t1 - normalised -  PalSAR bands')
    _debug_print_bands(t1)
    print('t2 - normalised -  RADARSAT-2 bands')
    _debug_print_bands(t2)

    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    change_mask = tf.convert_to_tensor(changemap[:,:,0], dtype=tf.bool)
    print(t1.shape)
    print(t2.shape)
    print(change_mask.shape)

    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reduce has been DISABLED")

    return t1, t2, change_mask


def _clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    temp = np.reshape(image, (-1, image.shape[-1]))

    limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    for i, limit in enumerate(limits):
        channel = temp[:, i]
        channel = tf.clip_by_value(channel, 0, limit)
        ma, mi = tf.reduce_max(channel), tf.reduce_min(channel)
        channel = 2.0 * ((channel) / (ma)) - 1
        temp[:, i] = channel

    return tf.reshape(tf.convert_to_tensor(temp, dtype=tf.float32), image.shape)


def _norm01(input_array, norm_type='global', min_cap=None, max_cap=None, min_cap_value=np.NaN, max_cap_value=np.NaN):
    """Normalize data.
    
    Parameters:
    norm_type:
        'band' - min and max of each variable (column) is 0 and 1
        'global' - min of array is 0, max is 1
    min_cap: Truncate values below this value to min_cap_value before normalizing
    max_cap: Truncate values above this value to max_cap_value before normalizing
    """
    
    # Ensure that original input is not modified
    output_array = np.array(input_array, copy=True)
    
    # Replace values outside envolope/cap with NaNs (or specifie value)
    if min_cap is not None:
        output_array[output_array   < min_cap] = min_cap_value
                   
    if max_cap is not None:
        output_array[output_array  > max_cap] = max_cap_value
    
    # Normalize data for selected normalization option
    if norm_type.lower() in ['global', 'all', 'set']:
        # Normalize to 0-1 (globally)
        output_array = input_array - np.nanmin(input_array)
        output_array = output_array/np.nanmax(output_array)
    elif norm_type.lower() in ['band', 'channel']:
        # Normalize to 0-1 for each channel (assumed to be last index)
        # Get shape of input
        input_shape = input_array.shape
        output_array = np.zeros(input_shape)
        # Normalize each channel
        for i_channel in range(0, input_shape[2]):
            output_array[:,:,i_channel] = input_array[:,:,i_channel] - np.nanmin(input_array[:,:,i_channel])
            output_array[:,:,i_channel] = output_array[:,:,i_channel]/np.nanmax(output_array[:,:,i_channel])
        
    return output_array


def _debug_print_bands(arr_print, chans_print= None):
    """Debug, print statistics for each band."""
    
    # Print all channels if no list is provided
    if chans_print is None:
        chans_print = range(0, arr_print.shape[2])
    
    # Print
    for i_chan in chans_print:
        print(np.min(arr_print[:,:,i_chan]), 
        np.mean(arr_print[:,:,i_chan]), np.max(arr_print[:,:,i_chan]))


def _training_data_generator(x, y, p, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    c_x, c_y = x.shape[2], y.shape[2]
    chs = c_x + c_y + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, chs, 1)

    data = tf.concat([x, y, p], axis=-1)

    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs]

    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
    )

    return gen, dtypes, shapes


DATASETS = {
    "Texas": _texas,
    "California": _california,
    "France": _france,
    "Italy": _italy,
    "UK": _uk,
    "Denmark": _denmark,
    "Polmak-LS5-S2": _polmak_ls5_s2,
    "Polmak-LS5-S2-warp": _polmak_ls5_s2_warp_align,
    "Polmak-LS5-S2-collocate": _polmak_ls5_s2_snap_collocate,
    "Polmak-LS5-S2-NDVI": _polmak_ls5_s2_ndvi,
    "Polmak-LS5-PGNLM_A": _polmak_ls5_pgnlma,
    "Polmak-LS5-PGNLM_C": _polmak_ls5_pgnlmc,
    "Polmak-LS5-PGNLM_A-stacked": _polmak_ls5_pgnlma_stacked,
    "Polmak-LS5-PGNLM_C-stacked": _polmak_ls5_pgnlmc_stacked,
    "Polmak-A2-S2": _polmak_a2_s2,
    "Polmak-A2-S2-collocate": _polmak_a2_s2_snap_collocate,
    "Polmak-Pal-RS2_010817-collocate": _polmak_pal_rs2_010817,
}
prepare_data = {
    "Texas": True,
    "California": True,
    "France": True,
    "Italy": False,
    "UK": True,
    "Denmark": False,
    "Polmak-LS5-S2": False,
    "Polmak-LS5-S2-warp": False,
    "Polmak-LS5-S2-collocate": False,
    "Polmak-LS5-S2-NDVI": False,
    "Polmak-LS5-PGNLM_A": False,
    "Polmak-LS5-PGNLM_C": False,
    "Polmak-LS5-PGNLM_A-stacked": False,
    "Polmak-LS5-PGNLM_C-stacked": False,
    "Polmak-A2-S2": False,
    "Polmak-A2-S2-collocate": False,
    "Polmak-Pal-RS2_010817-collocate": False,
}


def fetch_fixed_dataset(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    try:
        initial_cm = load_prior(name, x_im.shape[:2])
    except (FileNotFoundError, KeyError) as e:
        print("Evaluating and saving prior")
        initial_cm = evaluate_prior(name, x_im, y_im, **kwargs)
    cross_loss_weight = 1 - initial_cm
    cross_loss_weight -= tf.reduce_min(cross_loss_weight)
    cross_loss_weight /= tf.reduce_max(cross_loss_weight)

    tr_gen, dtypes, shapes = _training_data_generator(
        x_im, y_im, cross_loss_weight, patch_size
    )
    training_data = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
    training_data = training_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = [tf.expand_dims(tensor, 0) for tensor in [x_im, y_im, target_cm]]
    if not tf.test.is_gpu_available():
        dataset = [tf.image.central_crop(tensor, 0.1) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = shapes[0][-1], shapes[1][-1]

    return training_data, evaluation_data, (c_x, c_y)


def fetch_CGAN(name, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    ps = kwargs.get("patch_size")
    y_im, x_im, target_cm = DATASETS[name](prepare_data[name])
    if not tf.test.is_gpu_available():
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]
    chs = [tensor.shape[-1] for tensor in dataset]
    dataset = [remove_borders(tensor, ps) for tensor in dataset]
    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))
    dataset = [image_in_patches(tensor, ps) for tensor in dataset]
    tot_patches = dataset[0].shape[0]
    return dataset[0], dataset[1], evaluation_data, (chs[0], chs[1]), tot_patches


def fetch(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])
    if not tf.test.is_gpu_available():
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y = dataset[0], dataset[1]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")
        fetch_fixed_dataset(DATASET)
