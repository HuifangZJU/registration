import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import tifffile as tiff
import os
from PIL import Image
import time
import matplotlib.patches as patches
import scanpy as sc

def visualize_image_with_polygons(image_data, cell_boundaries):
    img_np = image_data[0,:,:]

    fig, ax = plt.subplots(figsize=(10, 10))
    # Optional: set figure size
    ax.imshow(img_np, origin='upper')  # origin='upper' matches (0,0) at top-left

    # Plot the boundaries
    for _, group in cell_boundaries.groupby("cell_id"):
        coords = group[["vertex_x", "vertex_y"]].values
        polygon = Polygon(coords, closed=True, fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(polygon)

    # Formatting
    ax.set_title("Cell Boundaries Overlayed on Tissue Image", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def normalize_to_save(img):
    # Check the min and max of the original image
    img_min = img.min()
    img_max = img.max()
    # print(f"Original Image Min: {img_min}, Max: {img_max}")

    # Normalize the image to [0, 255]
    # Formula: (pixel_value - min) / (max - min) * 255
    # Since we know min=0 and max=4469, this simplifies to:
    img_normalized = ((img.astype(np.float32)-img_min) / (img_max-img_min)) * 255.0

    # Convert back to an unsigned 8-bit integer type
    img_normalized = img_normalized.astype(np.uint8)
    return img_normalized


def load_ome_tiff_efficiently(ome_tiff_path, use_dask=False):
    """
    Efficiently loads an OME-TIFF file, either as a memory-mapped array or using Dask for lazy loading.

    Parameters:
        ome_tiff_path (str): Path to the OME-TIFF file.
        use_dask (bool): Whether to use Dask for lazy loading. Defaults to False.

    Returns:
        array-like: The loaded image data (NumPy or Dask array).
    """
    try:
        with tiff.TiffFile(ome_tiff_path) as ome_tiff:
            # Inspect metadata
            print("OME-TIFF Metadata:")
            print(ome_tiff.series[0].shape)  # Example of inspecting the shape of the first series

            # Access the desired series (e.g., the first one)
            series = ome_tiff.series[0]

            if use_dask:
                # Load as Dask array for lazy processing
                image_data = da.from_array(series.asarray(), chunks="auto")
            else:
                # Load as memory-mapped NumPy array for immediate access
                image_data = series.asarray(out="memmap")

            # Select the first slice if ndim > 2
            #if image_data.ndim > 2:
                #image_data = image_data[0, 0]

            return image_data
    except Exception as e:
        print(f"Error loading OME-TIFF: {e}")
        return None

def save_regional_cell_locations():
    cell_path = subfolder + '/cells.csv.gz'
    with open(subfolder + '/experiment.xenium', "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size") * 4

    cells = pd.read_csv(cell_path)
    cells['x_centroid'] = cells['x_centroid'] / pixel_size
    cells['y_centroid'] = cells['y_centroid'] / pixel_size

    crop_df = pd.read_csv(subfolder + '/crop_boxes.csv')
    crop_section = crop_df[['x1', 'y1', 'x2', 'y2']].values.tolist()

    regional_cells = []
    if not crop_section:
        cells['region_id'] = 0
        # cells['x_centroid'] = cells['x_centroid']
        # cells['y_centroid'] = cells['y_centroid']
        cells['x_centroid'] = cells['x_centroid'] * 4
        cells['y_centroid'] = cells['y_centroid'] * 4
        regional_cells.append(cells[['cell_id', 'region_id', 'x_centroid', 'y_centroid']])
        # regional_image = plt.imread(
        #     subfolder + "/morphology_focus/channel0_quarter.png")
        # plt.figure(figsize=(12, 8))
        # plt.imshow(regional_image)
        # plt.scatter(regional_cells[0]['x_centroid'], regional_cells[0]['y_centroid'], s=2)
        # plt.show()
    else:
        # Assign region ID and compute regional coordinates
        for region_id, (x1, y1, x2, y2) in enumerate(crop_section):
            # regional_image = plt.imread(
            #     subfolder + "/morphology_focus/regional_images" + f"/channel0_region_{region_id}_quarter.png")
            # Filter cells in this region
            mask = (
                    (cells['x_centroid'] >= x1) & (cells['x_centroid'] <= x2) &
                    (cells['y_centroid'] >= y1) & (cells['y_centroid'] <= y2)
            )
            region_cells = cells[mask].copy()

            # Compute regional coordinates
            region_cells['region_id'] = region_id
            region_cells['x_centroid'] = region_cells['x_centroid'] - x1
            region_cells['y_centroid'] = region_cells['y_centroid'] - y1
            region_cells['x_centroid'] = region_cells['x_centroid'] * 4
            region_cells['y_centroid'] = region_cells['y_centroid'] * 4

            regional_cells.append(region_cells[['cell_id', 'region_id', 'x_centroid', 'y_centroid']])

            # plt.figure(figsize=(12,8))
            # plt.imshow(regional_image)
            # plt.scatter(region_cells['x_centroid'],region_cells['y_centroid'],s=2)
            # plt.show()

    # # Concatenate all results
    regional_cells_df = pd.concat(regional_cells, ignore_index=True)
    # Save to CSV
    regional_cells_df.to_csv(subfolder + "/regional_cell_coordinates.csv", index=False)
    print(f"Saved regional cell data for {len(regional_cells_df)} cells to {sample_id}_regional_cell_coordinates.csv")

def merge_and_split_anndata(adata, cell_metadata_df, out_dir, sample_id):
    os.makedirs(out_dir, exist_ok=True)
    print(adata)


    # Ensure index consistency
    adata.obs['cell_id'] = adata.obs_names
    merged_df = pd.merge(
        adata.obs[['cell_id']],
        cell_metadata_df,
        how='inner',
        on='cell_id'
    )

    # Subset AnnData to matching cells
    adata_matched = adata[merged_df['cell_id'].values].copy()

    # Add the spatial metadata to AnnData obs
    for col in ['region_id', 'x_centroid', 'y_centroid']:
        adata_matched.obs[col] = merged_df[col].values

    # Split by region and save
    for region_id in sorted(adata_matched.obs['region_id'].unique()):
        region_data = adata_matched[adata_matched.obs['region_id'] == region_id].copy()
        print(region_data)
        test = input()
        file_path = os.path.join(out_dir, f"{sample_id}_{region_id}.h5ad")
        region_data.write(file_path)
        print(f"Saved: {file_path}")

def plot_gene_map():
    # Set gene to plot (use exact gene name from cell_gene_matrix.var_names)
    gene = 'CDKN1A'  # Example, replace with your gene of interest

    # Ensure gene exists
    if gene not in cell_gene_matrix.var_names:
        raise ValueError(f"Gene '{gene}' not found in matrix. Check spelling.")

    # Get expression values as Series (indexed by cell ID)
    gene_expr = cell_gene_matrix[:, gene].to_df()[gene]
    gene_expr.index.name = 'cell_id'

    # Merge expression with regional positions
    merged = cells_locations.merge(gene_expr, on='cell_id', how='inner')
    merged.rename(columns={gene: 'expression'}, inplace=True)

    # --- Plot each region separately ---
    regions = sorted(merged['region_id'].unique())
    for region in regions:
        region_data = merged[merged['region_id'] == region].copy()
        region_data['x_centroid'] = region_data['x_centroid'] / 2
        region_data['y_centroid'] = region_data['y_centroid'] / 2

        x_range = region_data['x_centroid'].max() - region_data['x_centroid'].min()
        y_range = region_data['y_centroid'].max() - region_data['y_centroid'].min()
        aspect_ratio = y_range / x_range

        # Scale the figure width, height accordingly
        fig_width = x_range / 500  # or any base width you like
        fig_height = fig_width * aspect_ratio

        plt.figure(figsize=(fig_width, fig_height))
        region_data['log_expression'] = np.log1p(region_data['expression'])  # log(1 + x)

        scfig = plt.scatter(
            region_data['x_centroid'],
            region_data['y_centroid'],
            c=region_data['log_expression'],
            cmap='viridis',
            s=2,
            alpha=0.7
        )
        plt.colorbar(scfig, label=f'log(1 + {gene}) intensity')
        # scfig = plt.scatter(
        #     region_data['x_centroid'],
        #     region_data['y_centroid'],
        #     c=region_data['expression'],
        #     cmap='viridis',
        #     s=2,
        #     alpha=0.7
        # )
        # plt.colorbar(scfig, label=f'{gene} intensity')
        plt.axis('equal')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        name = f'{gene} on  Sample: {sample_id} | Region: {region}'
        plt.title(name, fontsize=fig_width * 1.5)
        plt.tight_layout()
        # plt.show()
        out_path = os.path.join("/media/huifang/data/sennet/xenium/lay_out_figures", name + '.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print('saved')

root_path = "/media/huifang/data/sennet/xenium/"

dataset = open(root_path+'data_list.txt')
lines = dataset.readlines()
for i in range(len(lines)):
    line = lines[i].rstrip().split(' ')
    subfolder = os.path.join(root_path,line[0])
    if os.path.exists(os.path.join(subfolder,'outs')):
        subfolder = os.path.join(subfolder,'outs')
    sample_id = line[1]

    print(sample_id)
    # save_regional_cell_locations()
    cell_gene_matrix = sc.read_10x_h5(subfolder + '/cell_feature_matrix.h5')
    cell_path = subfolder + "/regional_cell_coordinates.csv"
    cells_locations = pd.read_csv(cell_path)

    merge_and_split_anndata(cell_gene_matrix, cells_locations, '/media/huifang/data/sennet/xenium/regional_data', sample_id)


















