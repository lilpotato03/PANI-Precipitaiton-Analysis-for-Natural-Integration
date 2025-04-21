import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_initialization import categorical_labels,band_cmaps
import numpy as np
import matplotlib.pyplot as plt

def plot_bands_one_by_one(smoothed_stack, band_names, region_mask, band_cmaps=band_cmaps,categorical_labels=categorical_labels,breaks=None):
    num_bands = smoothed_stack.keys()

    # for i, band in enumerate(band_names):
    #     data = smoothed_stack[band]

    #     if region_mask is not None:
    #         if region_mask.shape != data.shape:
    #             print(f'{band}: mask and stack image shape mismatch, skipping')
    #             continue
    #         else:
    #             data = np.where(region_mask == 1, data, np.nan)

    #     fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    #     inner_ax = fig.add_subplot(111, facecolor='white')

    #     cmap = band_cmaps.get(band, 'viridis')

    #     if band in categorical_labels and band in categorical_labels.keys():
    #         labels = categorical_labels[band]
    #         categories = sorted(labels.keys())
    #         cmap_obj = plt.get_cmap(cmap, len(categories))
    #         norm = BoundaryNorm(categories + [max(categories)+1], cmap_obj.N)
    #         img = inner_ax.imshow(data, cmap=cmap_obj, norm=norm)

    #         # Create legend
    #         legend_patches = [
    #             mpatches.Patch(color=cmap_obj(i), label=labels[val])
    #             for i, val in enumerate(categories) if val in labels
    #         ]
    #         inner_ax.legend(handles=legend_patches, loc='lower right', fontsize=8, frameon=True)
    #     elif breaks is not None and band in breaks:
    #         band_breaks = breaks[band]
    #         num_classes = len(band_breaks) + 1
    #         cmap_obj = plt.get_cmap(cmap, num_classes)
    #         adjusted_breaks = np.arange(num_classes) + 0.5
    #         norm = BoundaryNorm(adjusted_breaks, cmap_obj.N)
    #         img = inner_ax.imshow(data, cmap=cmap_obj, norm=norm) # Removed vmin and vmax here

    #         legend_patches = []
    #         labels = []
    #         lower_bound = np.nanmin(smoothed_stack[band][np.isfinite(smoothed_stack[band])])
    #         sorted_breaks = sorted(band_breaks)
    #         for j, upper_bound in enumerate(sorted_breaks):
    #             labels.append(f'[{lower_bound:.2f}, {upper_bound:.2f})')
    #             legend_patches.append(mpatches.Patch(color=cmap_obj(j), label=labels[-1]))
    #             lower_bound = upper_bound
    #         labels.append(f'[{lower_bound:.2f}, {np.nanmax(smoothed_stack[band][np.isfinite(smoothed_stack[band])]):.2f}]')
    #         legend_patches.append(mpatches.Patch(color=cmap_obj(num_classes - 1), label=labels[-1]))

    #         inner_ax.legend(handles=legend_patches, loc='lower right', fontsize=8, frameon=True)
    #     else:
    #         img = inner_ax.imshow(data, cmap=cmap)
    #         cbar = fig.colorbar(img, ax=inner_ax, shrink=0.8, orientation='vertical', pad=0.02)
    #         cbar.ax.tick_params(labelsize=8)

    #     inner_ax.set_title(f"{band}", fontsize=14)
    #     inner_ax.axis('off')

    #     for spine in inner_ax.spines.values():
    #         spine.set_edgecolor('black')
    #         spine.set_linewidth(2)

    #     plt.tight_layout(pad=2.0)
    #     plt.show()
    for i, band in enumerate(band_names):
        data = smoothed_stack[band]

        if region_mask is not None:
            data = np.where(region_mask == 1, data, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        inner_ax = fig.add_subplot(111, facecolor='white')

        cmap_plot = band_cmaps.get(band, 'viridis')  # Colormap for the plot itself
        img = inner_ax.imshow(data, cmap=cmap_plot)

        if band in categorical_labels:
            labels = categorical_labels[band]
            categories = sorted(labels.keys())
            cmap_legend = plt.cm.get_cmap(cmap_plot, len(categories))
            norm = BoundaryNorm(categories + [max(categories)+1], cmap_legend.N)
            img = inner_ax.imshow(data, cmap=cmap_legend, norm=norm)
            legend_patches = [mpatches.Patch(color=cmap_legend(i), label=labels[val]) for i, val in enumerate(categories) if val in labels]
            inner_ax.legend(handles=legend_patches, loc='lower right', fontsize=8, frameon=True)

        elif breaks is not None and band in breaks:
            band_breaks = breaks[band]
            num_classes = len(band_breaks) + 1
            cmap_legend = plt.cm.get_cmap(cmap_plot, num_classes)  # Colormap for the legend

            legend_patches = []
            labels = []
            sorted_breaks = sorted(band_breaks)
            lower_bound = np.nanmin(smoothed_stack[band][np.isfinite(smoothed_stack[band])])

            for j, upper_bound in enumerate(sorted_breaks):
                labels.append(f'[{lower_bound:.2f}, {upper_bound:.2f})')
                legend_patches.append(mpatches.Patch(color=cmap_legend(j), label=labels[-1]))
                lower_bound = upper_bound

            labels.append(f'[{lower_bound:.2f}, {np.nanmax(smoothed_stack[band][np.isfinite(smoothed_stack[band])]):.2f}]')
            legend_patches.append(mpatches.Patch(color=cmap_legend(len(sorted_breaks)), label=labels[-1]))

            inner_ax.legend(handles=legend_patches, loc='lower right', fontsize=8, frameon=True)

        else:
            cbar = fig.colorbar(img, ax=inner_ax, shrink=0.8, orientation='vertical', pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        inner_ax.set_title(f"{band}", fontsize=14)
        inner_ax.axis('off')
        for spine in inner_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.tight_layout(pad=2.0)
        plt.show()