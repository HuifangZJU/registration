import tifffile as  tiff
from matplotlib import pyplot as plt
import scanpy as sc
import pandas as pd
import os
from itertools import islice
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
# img = tiff.imread("/media/huifang/data/sennet/codex/ reg001_X01_Y01_Z01.tif")
# for i in range(img.shape[0]):
#     plt.imshow(img[i,:,:])
#     plt.show()

def save_per_slide():
    fixed_subplot_height = 4  # inch height of each row
    n_rows = 5  # fixed number of rows
    top_padding = 0.05  # reserve 10% of vertical space for title

    n_skip = 1  # number of slides to skip
    for slide_name, slide_adata in islice(adata_by_slide.items(), n_skip, None):
        # print(slide_name)
        # continue
        print(f"Plotting {slide_name}...")

        # Step 1: Gather regions and aspect ratios
        regions = slide_adata.obs['unique_region'].unique()
        region_infos = []
        for region in regions:
            data = slide_adata.obs[slide_adata.obs['unique_region'] == region]
            width = data['x'].max() - data['x'].min()
            height = data['y'].max() - data['y'].min()
            aspect = width / height if height > 0 else 1.0
            region_infos.append({'name': region, 'aspect': aspect, 'data': data})

        # Step 2: Sort by aspect and balance across fixed rows
        region_infos.sort(key=lambda x: -x['aspect'])
        row_assignments = [[] for _ in range(n_rows)]
        row_widths = [0] * n_rows
        for region in region_infos:
            i = row_widths.index(min(row_widths))
            row_assignments[i].append(region)
            row_widths[i] += region['aspect']

        # Step 3: Compute figure size
        max_row_width = max(row_widths)
        fig_height = fixed_subplot_height * n_rows / (1 - top_padding)  # increase to fit title
        fig_width = max_row_width * fixed_subplot_height - 5
        # print(fig_width)
        # print( fig_height)
        # test = input()
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Step 4: Plot subplots
        for row_idx, row in enumerate(row_assignments):
            total_width = sum(r['aspect'] for r in row)
            left_pos = 0
            for region in row:
                norm_width = region['aspect'] / total_width
                ax = fig.add_axes([
                    left_pos,  # left
                    (1 - top_padding) * (n_rows - row_idx - 1) / n_rows,  # bottom
                    norm_width,  # width
                    (1 - top_padding) / n_rows  # height
                ])
                left_pos += norm_width

                ax.scatter(region['data']['x'], region['data']['y'], s=1, alpha=0.5)
                ax.set_title(f"Region: {region['name']}", fontsize=10)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

        # Step 5: Manually add title above all plots
        fig.text(0.5, 1 - top_padding / 2, f"Slide: {slide_name}", ha='center', va='center', fontsize=16)
        # plt.show()
        # Save
        safe_name = slide_name.replace("/", "_").replace(" ", "_")
        out_path = os.path.join("/media/huifang/data/sennet/codex/layout_figures", f"{safe_name}_region_layout.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print('saved')
        test = input()


def save_per_region():
    for region in slide_adata.obs['unique_region'].unique():
        region_data = slide_adata.obs[slide_adata.obs['unique_region'] == region]

        # Get x/y ranges from cell centers
        x_min, x_max = region_data['x'].min(), region_data['x'].max()
        y_min, y_max = region_data['y'].min(), region_data['y'].max()
        width = x_max - x_min
        height = y_max - y_min

        # Set base size and compute aspect-matching figsize
        base_size = 25  # scale this as needed
        aspect_ratio = height / width
        figsize = (base_size, base_size * aspect_ratio)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(region_data['x'], region_data['y'], s=1, alpha=0.8)
        ax.set_title(f"{slide_name} - {region}")
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        # plt.show()

        # Safe file name
        safe_slide = slide_name.replace("/", "_").replace(" ", "_")
        safe_region = str(region).replace("/", "_").replace(" ", "_")
        filename = f"{safe_slide}__{safe_region}.png"
        out_path = os.path.join("/media/huifang/data/sennet/codex/layout_figures", filename)

        plt.savefig(out_path, dpi=300)
        plt.close()

def save_to_file(adata_by_slide):
    for slide_name, slide_adata in adata_by_slide.items():
        # Optional: sanitize the slide_name to ensure it's safe as a filename
        safe_slide_name = slide_name.replace("/", "_").replace(" ", "_")

        # Define the file path
        file_path = os.path.join("/media/huifang/data/sennet/codex/", f"{safe_slide_name}.h5ad")

        # Save the AnnData object
        slide_adata.write(file_path)


def split_and_save_by_slide_and_region(adata, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Iterate through unique slide names
    for slide in adata.obs['slide_name'].unique():
        slide_data = adata[adata.obs['slide_name'] == slide]

        # Within each slide, split by unique region
        for region in slide_data.obs['unique_region'].unique():
            region_data = slide_data[slide_data.obs['unique_region'] == region]

            # Construct filename and save
            file_name = f"{slide}_{region}.h5ad"
            file_path = os.path.join(out_dir, file_name)
            region_data.write(file_path)
            print(f"Saved: {file_path}")

adata = sc.read_h5ad("/media/huifang/data/sennet/codex/20250603_sennet_annotated_updated_donormeta.h5ad")
# split_and_save_by_slide_and_region(adata,'/media/huifang/data/sennet/codex/regional_data/')



sample_id = adata.obs['sample'].unique()
# print(adata)
slide_names = adata.obs['slide_name'].unique()
# Create a dictionary of AnnData objects, split by 'slide_name'
adata_by_slide = {name: adata[adata.obs['slide_name'] == name].copy() for name in slide_names}
# save_to_file(adata_by_slide)
n_skip = 0  # number of slides to skip
for slide_name, slide_adata in islice(adata_by_slide.items(), n_skip, None):
    print(slide_name)

    # print(slide_adata)
    regions = slide_adata.obs['unique_region'].unique().tolist()
    for selected_region in regions:
        print(selected_region)
        # 'sample' --> xenium sample id
        # Filter obs by region
        subset = slide_adata.obs[slide_adata.obs['unique_region']==selected_region]
        sample_id = subset['sample'].unique().tolist()
        if len(sample_id)>1:
            print('Error in smaple id!')
           # Choose marker
        # marker = 'p53'
        # markers = ['DAPI', 'p53', 'p16', 'pH2AX', 'HSP47', 'PD1', 'PDL1', 'CD32b', 'SPP1', 'CD23']
        markers=['p16']
        for marker in markers:
            # Plot
            x_range = subset['x'].max() - subset['x'].min()
            y_range = subset['y'].max() - subset['y'].min()
            aspect_ratio = y_range / x_range

            # Scale the figure width, height accordingly
            fig_width = x_range / 500  # or any base width you like
            fig_height = fig_width * aspect_ratio


            plt.figure(figsize=(fig_width, fig_height))
            sc = plt.scatter(
                subset['x'],
                subset['y'],
                c=subset[marker],
                cmap='viridis',
                s=2,
                alpha=0.7
            )
            plt.colorbar(sc, label=f'{marker} intensity')
            plt.axis('equal')
            name = f'{marker} on Slide: {slide_name} | Region: {selected_region} | Sample: {sample_id[0]}'
            plt.title(name,fontsize=fig_width*1.5)
            plt.tight_layout()
            # plt.show()
            out_path = os.path.join("/media/huifang/data/sennet/codex/layout_figures", name+'.png')
            plt.savefig(out_path, dpi=300)
            plt.close()
            # print('saved')
            # test = input()








