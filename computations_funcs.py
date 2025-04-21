import numpy as np
from scipy.ndimage import zoom
from jenkspy import JenksNaturalBreaks
from sklearn.cluster import KMeans
from data_initialization import categorical_labels
import pandas as pd

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
        if arr.ndim != 2:
            print(f"Skipping {band} due to shape: {arr.shape}")
            continue
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

def generate_probability_mask(data,band_names,weights):
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
    if data is None:
        raise ValueError("data should not be None")
    bands = []
    wts = []

    for id,band in enumerate(band_names):
        if band in band_names:
            band_data=(np.array(data[id], dtype=np.float32))
            wts.append(weights.get(band, 0))
            min_val = np.nanmin(band_data)
            max_val = np.nanmax(band_data)
            if max_val > min_val:
                normalized_band = (band_data - min_val) / (max_val - min_val + 1e-8)
            else:
                normalized_band = np.zeros_like(band_data)  # Handle constant bands
            bands.append(normalized_band)

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

def normalize_layer(array):
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))


def quantile_bins(data, n_classes=5):
    flat = data[~np.isnan(data)].flatten()
    breaks = np.percentile(flat, np.linspace(0, 100, n_classes + 1))
    tiers = np.digitize(data, breaks[1:], right=False)
    return tiers, breaks



def kmeans_bins(data, n_classes=5):
    # Flatten the data and remove NaN values
    flat = data[~np.isnan(data)].flatten().reshape(-1, 1)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_classes, n_init='auto', random_state=42).fit(flat)
    
    # Sort cluster centers
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Create the breaks array
    breaks = [float(np.min(flat))] + \
            [float((centers[i] + centers[i+1]) / 2) for i in range(len(centers) - 1)] + \
            [float(np.max(flat))]
    breaks = np.array(breaks).flatten()
    # Ensure breaks is a numpy array for consistency
    flat_data = data.flatten()
    tiers = np.digitize(flat_data, breaks[1:], right=False)
    tiers = tiers.reshape(data.shape)
    # Use np.digitize to assign data to the correct tier
    
    return tiers, breaks

def convert_smooth_data_to_bins(smoothed_stack,band_names,n_classes=5,categorical_bands=categorical_labels.keys()):
    tiers = {}
    breaks = {}
    for id,band in enumerate(band_names):
        if band not in categorical_bands:
            tiers[band],breaks[band] = kmeans_bins(smoothed_stack[:,:,id],n_classes=n_classes)
        else:
            tiers[band]=smoothed_stack[:,:,id]
    return tiers,breaks

def quantile_bins(data, n_classes=5):
    flat_data = data[~np.isnan(data)].flatten()
    breaks = np.percentile(flat_data, np.linspace(0, 100, n_classes + 1))
    unique_breaks = np.unique(breaks)  # Remove duplicate breaks if any

    tiers = np.digitize(data, unique_breaks[1:], right=False)

    return tiers, unique_breaks


def create_balanced_csv(stacked_array, level_mask, prob_mask, band_names, region_mask, num_points=1000, seed=42,name='PANI_Dataset'):
    file_name=f'{name}.csv'
    np.random.seed(seed)
    h, w, n_bands = stacked_array.shape

    flat_data = stacked_array.reshape(-1, n_bands)
    flat_levels = level_mask.flatten()  
    flat_probs = prob_mask.flatten()
    flat_mask = region_mask.flatten()

    valid_indices = np.where(flat_mask == 1)[0]

    flat_data = flat_data[valid_indices]
    flat_levels = flat_levels[valid_indices]
    flat_probs = flat_probs[valid_indices]
    print(f"Valid flat_level:{np.unique(flat_levels)}")
    print(flat_data.shape)
    samples_per_class = num_points // 5
    sampled_rows = []

    for level in range(1, 6): 
        level_indices = np.where(flat_levels == level)[0]

        if len(level_indices) < samples_per_class:
            print(f"⚠️ Not enough samples for level {level}. Found: {len(level_indices)}")
            sampled = level_indices
        else:
            sampled = np.random.choice(level_indices, size=samples_per_class, replace=False)

        for idx in sampled:
            row = list(flat_data[idx]) + [flat_levels[idx]]
            sampled_rows.append(row)

    col_names = band_names + ['level']
    df = pd.DataFrame(sampled_rows, columns=col_names)

    df.to_csv(file_name, index=False)
    print(f"✅ Saved to '{file_name}'")

    return df

