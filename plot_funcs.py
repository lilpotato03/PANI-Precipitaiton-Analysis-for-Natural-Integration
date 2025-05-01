import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_initialization import categorical_labels,band_cmaps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bands_one_by_one(smoothed_stack, band_names, region_mask, band_cmaps=band_cmaps,categorical_labels=categorical_labels,breaks=None):
    num_bands = smoothed_stack.keys()
    for i, band in enumerate(band_names):
        data = smoothed_stack[band]

        if region_mask is not None:
            data = np.where(region_mask == 1, data, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        inner_ax = fig.add_subplot(111, facecolor='white')

        cmap_plot = band_cmaps.get(band, 'viridis')  # Colormap for the plot itself

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
            num_classes = len(band_breaks)-1
            cmap_legend = plt.cm.get_cmap(cmap_plot, num_classes)
            cmap_colors = cmap_legend(np.arange(num_classes))
            cmap = ListedColormap(cmap_colors)
            masked_data = np.where(data == 0, np.nan, data - 1)
            fimg = inner_ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=num_classes - 1)
            legend_patches = []
            for i in range(num_classes):
                lower = band_breaks[i]
                upper = band_breaks[i+1]
                label = f"[{lower:.2f}, {upper:.2f})" if i < num_classes - 1 else f"[{lower:.2f}, {upper:.2f}]"
                patch = mpatches.Patch(color=cmap_colors[i], label=label)
                legend_patches.append(patch)
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distribution(stack, bands, categorical_bands=[],mask_value=99999999,bins=5):
    for idx, band in enumerate(bands):
        image = stack[:, :, idx]
        flat_data = image[~np.isnan(image)].flatten()
        print('number of pixels of data',len(image.flatten()))
        print('number of pixels of flat data',len(flat_data))
        plt.figure(figsize=(8, 4))
        plt.title(f'{band} Value Distribution')
        total = len(flat_data)

        if band in categorical_bands:
            classes, counts = np.unique(flat_data, return_counts=True)
            bars=plt.bar(classes, counts)
            plt.xlabel('Category')
            plt.ylabel('Count')
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                percentage = (count / total) * 100
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{count} ({percentage:.2f}%)', ha='center')
        else:
            hist=sns.histplot(flat_data, bins=bins, kde=False)
            plt.xlabel('Value')
            plt.ylabel('Count')
            for patch in hist.patches:
                height = patch.get_height()
                percentage = (height / total) * 100
                plt.text(patch.get_x() + patch.get_width() / 2, height + 0.01,f'{int(height)} ({percentage:.2f}%)', ha='center')

        plt.tight_layout()
        plt.show()
