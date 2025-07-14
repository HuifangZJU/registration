import scanpy as sc
import pandas as pd
import seaborn as sns
import collections
import scipy.sparse as sp_sparse
import tables
import argparse
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
Image.MAX_IMAGE_PIXELS = None

from numba import jit


@jit
def filling_maps(featuremaps,featuredata,pixel_u,pixel_v,img_height,img_width):
    for d in range(featuredata.shape[1]):
        feature = featuredata[:, d]
        for u, v, fe in zip(pixel_u, pixel_v, feature):
            featuremaps[u,v,d] = fe
    return featuremaps

def showfeaturechanels(features,pixel_v,pixel_u,img=np.array([]),crop=True):
    if crop:
        vmin = np.amin(pixel_v)
        vmax = np.amax(pixel_v)
        umin = np.amin(pixel_u)
        umax = np.amax(pixel_u)
    else:
        vmin = 1
        vmax = img.shape[0]
        umin = 1
        umax= img.shape[1]

    f, axarr = plt.subplots(3, 4, figsize=(9, 6))
    f.tight_layout()
    if not np.any(img):
        axarr[0, 0].imshow(features[vmin:vmax, umin:umax, 11])
    else:
        axarr[0, 0].imshow(img[vmin:vmax, umin:umax, :])
        # axarr[0, 1].imshow(img[vmin:vmax, umin:umax, 1], cmap='Greens')
        # axarr[0, 2].imshow(img[vmin:vmax, umin:umax, 2], cmap='Blues')
    axarr[0, 1].matshow(features[vmin:vmax, umin:umax, 0])
    axarr[0, 2].matshow(features[vmin:vmax, umin:umax, 1])
    axarr[1, 0].matshow(features[vmin:vmax, umin:umax, 2])
    axarr[1, 1].matshow(features[vmin:vmax, umin:umax, 3])
    axarr[1, 2].matshow(features[vmin:vmax, umin:umax, 4])
    axarr[2, 0].matshow(features[vmin:vmax, umin:umax, 5])
    axarr[2, 1].matshow(features[vmin:vmax, umin:umax, 6])
    axarr[2, 2].matshow(features[vmin:vmax, umin:umax, 7])
    axarr[0, 3].matshow(features[vmin:vmax, umin:umax, 8])
    axarr[1, 3].matshow(features[vmin:vmax, umin:umax, 9])
    axarr[2, 3].matshow(features[vmin:vmax, umin:umax, 10])
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', type=str, default='/media/huifang/data/registration/mouse/anterior_v1/', help='rootpath to data')
opt = parser.parse_args()

#adata: spots are observation, genes are variables
adata = sc.read_visium(opt.rootpath,count_file='filtered_matrix.h5')
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
##filter out data by number_of_counts
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20]
sc.pp.filter_genes(adata, min_cells=10)
##normalize reads
sc.pp.normalize_total(adata, inplace=True)
## detect highly-variable genes
gene_feature_dimension = 2000
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=gene_feature_dimension)
highly_variable = adata.var["highly_variable"]
highly_variable_flag = highly_variable.values
highly_variable_data = adata[:, highly_variable_flag]
feature_data = highly_variable_data.X.toarray()
# print(highly_data.uns['spatial'])
[library_id]=highly_variable_data.uns['spatial'].keys()


img = highly_variable_data.uns['spatial'][library_id]['images']['lowres']
img_height = img.shape[0]
img_width = img.shape[1]

low_res_scale = highly_variable_data.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']
uv = highly_variable_data.obsm['spatial'].toarray() * low_res_scale
uv = uv.astype(int)

pixel_u = uv[:,1]
pixel_v = uv[:,0]

featuremaps = np.zeros([img_height,img_width,gene_feature_dimension])
featuremaps = filling_maps(featuremaps,feature_data,pixel_u,pixel_v,img_height,img_width)



showfeaturechanels(featuremaps,pixel_v,pixel_u,img,False)
# showfeaturechanels(countmaps,spot_y,spot_x)

##########visualization##################


# for i in range(gene_feature_dimension):
#     featuremap = featuremaps[:,:,i]
#     countmap = countmaps[:,:,i]
#
#     f, axarr = plt.subplots(1, 3)
#     axarr[0].imshow(img[vmin:vmax,umin:umax,:])
#     axarr[1].matshow(featuremap[vmin:vmax,umin:umax,])
#     axarr[2].matshow(countmap)
#     plt.show()


