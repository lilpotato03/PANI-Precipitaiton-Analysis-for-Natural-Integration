import numpy as np
from scipy.ndimage import zoom


def fill_depressions(dem):
    filled = dem.focal_min(2).unmask(dem)
    return filled


def resize_img(img, target_size=512):
    height, width = img.shape
    if height < width:
        scale_factor = target_size / height
    else:
        scale_factor = target_size / width
        resized_height = int(height * scale_factor)
        resized_width = int(width * scale_factor)
        resized_img = zoom(
        img, (resized_height / height, resized_width / width), order=0
        )  # Nearest-neighbor for img
    return resized_img


def smooth_stack_from_array_dict(array_dict, target_size=512, categorical_bands=None):
    band_data = array_dict["properties"]
    band_names = list(band_data.keys())
    resized_bands = []

    for band in band_names:
        arr = np.array(band_data[band])

        # Compute target scale factor for the shortest side
        height, width = arr.shape
        if height < width:
            scale_factor = target_size / height
        else:
            scale_factor = target_size / width

        # Compute new dimensions
        resized_height = int(height * scale_factor)
        resized_width = int(width * scale_factor)

        # Determine interpolation method
        interp_order = (
            0 if categorical_bands and band in categorical_bands else 1
        )  # NN for categorical, linear for continuous

        # Resize
        resized = zoom(
            arr, (resized_height / height, resized_width / width), order=interp_order
        )

        # Round categorical bands back to integers
        if interp_order == 0:
            resized = np.round(resized)

        resized_bands.append(resized)

    # Stack all bands into final array
    stacked = np.stack(resized_bands, axis=-1)  # Shape: (H, W, num_bands)
    return stacked, band_names

import numpy as np
import matplotlib.pyplot as plt

def generate_probability_mask(smoothed_stack,band_names,weights):
    """
    Generate a probability mask from weighted bands in array_dict,
    and classify it into 5 levels: [0-0.2), [0.2-0.4), ..., [0.8-1.0].

    Parameters:
    - array_dict (dict): Keys are band names, values are 2D numpy arrays.
    - weights (dict): Band weights, will be normalized.

    Returns:
    - prob_mask: Normalized probability mask (2D float array in [0, 1])
    - level_mask: Integer levels from 1 to 5 corresponding to probability bins
    - weighted_sum: Raw weighted sum before normalization
    """
    if band_names is None:
        raise ValueError("band_names should not be None")
    if weights is None:
        raise ValueError("weights should not be None")
    if smoothed_stack is None:
        raise ValueError("smoothed_stack should not be None")
    bands = []
    wts = []

    for id,band in enumerate(band_names):
        if band in band_names:
            bands.append(np.array(smoothed_stack[:,:,id], dtype=np.float32))
            wts.append(weights.get(band, 0))

    stacked = np.stack(bands, axis=-1)
    wts = np.array(wts, dtype=np.float32)
    wts /= wts.sum()  # Normalize weights

    print(f'stacked shape: {stacked.shape}')
    print(f'wts shape: {wts.shape}')

    weighted_sum = np.tensordot(stacked, wts, axes=([2], [0]))

    # Normalize to [0, 1]
    min_val, max_val = weighted_sum.min(), weighted_sum.max()
    prob_mask = (weighted_sum - min_val) / (max_val - min_val + 1e-8)

    # Classify into 5 levels: values from 1 to 5
    level_mask = np.digitize(prob_mask, bins=[0.2, 0.4, 0.6, 0.8, 1.0])

    # Add 1 to make levels 1 to 5
    level_mask += 1

    return prob_mask, level_mask, weighted_sum
