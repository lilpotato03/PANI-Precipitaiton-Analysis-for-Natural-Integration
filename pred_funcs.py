from data_collection import (
    get_state_districs,
    get_region_for_districts,
    generate_mask_for_region,
    get_data_for_districts,
)
from computations_funcs import smooth_stack_from_array_dict,convert_smooth_data_to_bins,generate_probability_mask
from data_initialization import categorical_labels,mcdm_weights
import numpy as np
mask_value = 999999


def get_pred_data():
    state = ""
    while state == "":
        state = input("Please Enter a state name:")
    state_source, districts = get_state_districs(state)
    print(f"List of Districts in {state} are:\n{districts}")
    print("\n")
    selected_districts_unfiltered = input(
        "Please Enter the name of the Districts separated by comma:\n"
    )
    selected_districts = selected_districts_unfiltered.split(",")
    selected_districts = [district.strip() for district in selected_districts]

    print(f"Selected Districts are:\n{selected_districts}")
    region, scale, region_self_mask = get_region_for_districts(
        state_source, selected_districts
    )
    mask_projection, region_mask = generate_mask_for_region(region, scale)
    array_dict = get_data_for_districts(
        region, scale, mask_projection, region_self_mask, mask_value
    )
    smoothed_stack, band_names = smooth_stack_from_array_dict(
        array_dict, categorical_bands=categorical_labels.keys()
    )
    tiers,breaks=convert_smooth_data_to_bins(smoothed_stack,band_names,n_classes=5)
    for band in tiers.keys():
        tiers[band] = np.nan_to_num(tiers[band])
    return tiers,region_mask

def predict_image(data,region_mask,mlp,scaler):
    data=np.array([x for x in data.values()])
    C, H, W = data.shape
    data = np.transpose(data, (1, 2, 0)) 
    mask_flat = region_mask.flatten() == 1
    data_flat = data.reshape(-1, C)
    X_masked = data_flat[mask_flat]
    X_scaled = scaler.transform(X_masked)
    y_pred_masked = mlp.predict(X_scaled)
    prediction_full = np.full((H * W,), -1)
    prediction_full[mask_flat] = y_pred_masked
    predicted_image = prediction_full.reshape(H, W)
    return predicted_image

