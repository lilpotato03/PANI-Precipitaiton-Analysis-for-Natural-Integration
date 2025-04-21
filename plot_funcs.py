import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_initialization import categorical_labels,band_cmaps
import numpy as np
import matplotlib.pyplot as plt

def plot_bands_one_by_one(stacked_array, band_names, region_mask, band_cmaps=band_cmaps,categorical_labels=categorical_labels):
    num_bands = stacked_array.shape[-1]

    for i, band in enumerate(band_names):
        data = stacked_array[:, :, i]

        if region_mask is not None:
            if region_mask.shape != data.shape:
                print(f'{band}: mask and stack image shape mismatch, skipping')
                continue
            else:
                data = np.where(region_mask == 1, data, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        inner_ax = fig.add_subplot(111, facecolor='white')

        cmap = band_cmaps.get(band, 'viridis')

        if band in categorical_labels and band in categorical_labels.keys():
            labels = categorical_labels[band]
            categories = sorted(labels.keys())
            cmap_obj = plt.get_cmap(cmap, len(categories))
            norm = BoundaryNorm(categories + [max(categories)+1], cmap_obj.N)
            img = inner_ax.imshow(data, cmap=cmap_obj, norm=norm)

            # Create legend
            legend_patches = [
                mpatches.Patch(color=cmap_obj(i), label=labels[val])
                for i, val in enumerate(categories) if val in labels
            ]
            inner_ax.legend(handles=legend_patches, loc='lower right', fontsize=8, frameon=True)

        else:
            img = inner_ax.imshow(data, cmap=cmap)
            cbar = fig.colorbar(img, ax=inner_ax, shrink=0.8, orientation='vertical', pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        inner_ax.set_title(f"{band}", fontsize=14)
        inner_ax.axis('off')

        for spine in inner_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        plt.tight_layout(pad=2.0)
        plt.show()