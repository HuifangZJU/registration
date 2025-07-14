import pandas as pd
import scanpy as sc
import glob
import os
df = pd.read_csv('/media/huifang/data/sennet/xenium/ct.csv')
# Rename column for convenience
# Rename the column for convenience
df = df.rename(columns={'Unnamed: 0': 'full_id'})

# Split 'full_id' into sample_id and cell_id
split_cols = df['full_id'].str.split('_', n=1, expand=True)
split_cols.columns = ['sample_id', 'cell_id']
# Clean sample_id by removing the last '-' if present
split_cols['sample_id'] = split_cols['sample_id'].str.replace(r'-(?=[^-\s]*$)', '', regex=True)
# Show cleaned unique sample_ids for double check
unique_sample_ids = split_cols['sample_id'].unique()
# print(f"Total unique cleaned sample_ids: {len(unique_sample_ids)}")
# print(unique_sample_ids)
# test = input()
# Add the new columns to the original DataFrame
df['sample_id'] = split_cols['sample_id']
df['cell_id'] = split_cols['cell_id']

# Group by sample_id
grouped = dict(tuple(df.groupby('sample_id')))

# OPTIONAL: Save each group to separate CSV
for sample_id, group in grouped.items():
    print(sample_id)
    folder_path = '/media/huifang/data/sennet/registered_data/'
    pattern = os.path.join(folder_path, f'xenium_{sample_id}_*.h5ad')
    file_list = glob.glob(pattern)
    for file in file_list:
        print(file)
        xenium_region_data = sc.read_h5ad(file)

        # Ensure unique mapping
        group_indexed = group.set_index('cell_id')
        xenium_region_data.obs.set_index('cell_id', inplace=True)

        # Map 'ct' to 'cell_type', based on shared cell_id
        if xenium_region_data.obs.index.isin(group_indexed.index).any():
            xenium_region_data.obs['cell_type'] = group_indexed['ct'].reindex(xenium_region_data.obs.index)
        else:
            print(f"Warning: No matching cell_id found in group for file {file}")
            xenium_region_data.obs['cell_type'] = None

        # Reset index if needed
        xenium_region_data.obs.reset_index(inplace=True)

        # Save the modified file
        newpath = file.replace('.h5ad', '_with_ct.h5ad')
        xenium_region_data.write(newpath)
        print('saved')




    # group.to_csv(f"{sample_id}_celltypes.csv", index=False)