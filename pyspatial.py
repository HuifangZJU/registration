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
CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])


@jit
def filling_maps(featuremaps,filtered_matrix_h5,pixel_v,pixel_u,img_height,img_width):
    for d in range(filtered_matrix_h5.shape[0]):
        featuredata = filtered_matrix_h5[d, :]
        for v, u, fe in zip(pixel_v, pixel_u, featuredata):
            featuremaps[v,u,d] = fe
    return featuremaps

# get non_zero_percentage for each gene, dimension means gene id
# @jit
def get_non_zero_flags(matrix,non_zero_matrix):
    for dimension in range(matrix.shape[0]):
        feature_per_gene = matrix[dimension,:]
        indices = np.where(feature_per_gene > 0)
        non_zero_percentage = len(indices[0]) / matrix.shape[1]
        non_zero_matrix[dimension] = non_zero_percentage
    return non_zero_matrix

def densefilter(matrix,threhold):
    non_zero_matrix = np.zeros([matrix.shape[0],1])
    non_zero_matrix= get_non_zero_flags(matrix,non_zero_matrix)
    denseindices = np.where(non_zero_matrix>threhold)
    denseindices = denseindices[0]
    matrix = matrix[denseindices,:]
    return matrix

def get_matrix_from_h5(filename, threshold=0.9):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
        matrix = matrix.toarray()
        matrix = densefilter(matrix,threshold)
        return matrix
        ''' 
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            print(key)
        test = input()
        for key in tag_keys:
            print(key)
            feature_ref[key] = getattr(feature_group, key).read()
         
        return CountMatrix(feature_ref, barcodes, matrix)
        '''


def get_indices_from_file(filename,scale=1):
    indices = np.loadtxt(open(filename,"rb"),dtype=int, delimiter=",",usecols=(1,2,3,4,5))
    in_area = indices[:,0] == 1
    spot_y = indices[in_area, 1]
    spot_x = indices[in_area, 2]
    pixel_v = indices[in_area, 3] * scale
    pixel_u = indices[in_area, 4] * scale
    return spot_x, spot_y, pixel_u.astype(int), pixel_v.astype(int)

def get_res_scale(scale_file, scale_name):
    with open(scale_file, 'r') as f:
        data = json.load(f)
    return data[scale_name]


def get_images_from_file(filename):
    he_img = Image.open(filename)
    he_array = np.array(he_img)
    return he_array
def regularize_spot_scale(pixel_v,pixel_u,spot_y,spot_x):
    vmin = np.amin(pixel_v)
    vmax = np.amax(pixel_v)
    umin = np.amin(pixel_u)
    umax = np.amax(pixel_u)

    ymin = np.amin(spot_y)
    ymax = np.amax(spot_y)
    xmin = np.amin(spot_x)
    xmax = np.amax(spot_x)

    spot_scale_y = (vmax-vmin)/(ymax-ymin)
    spot_scale_x = (umax - umin)/(xmax - xmin)
    if spot_scale_y>spot_scale_x:
        spot_y = spot_y * spot_scale_y/spot_scale_x
        spot_y = spot_y.astype(int)
    else:
        spot_x = spot_x * spot_scale_x / spot_scale_y
        spot_x = spot_x.astype(int)
    return  spot_y, spot_x

def showfeaturechanels(features,pixel_v,pixel_u,img=np.array([])):
    vmin = np.amin(pixel_v)
    vmax = np.amax(pixel_v)
    umin = np.amin(pixel_u)
    umax = np.amax(pixel_u)
    f, axarr = plt.subplots(3, 4, figsize=(25, 20))
    f.tight_layout()
    if not np.any(img):
        axarr[0, 0].imshow(features[vmin:vmax, umin:umax, 9])
        axarr[0, 1].imshow(features[vmin:vmax, umin:umax, 10])
        axarr[0, 2].imshow(features[vmin:vmax, umin:umax, 11])
    else:
        axarr[0, 0].imshow(img[vmin:vmax, umin:umax, 0], cmap='Reds')
        axarr[0, 1].imshow(img[vmin:vmax, umin:umax, 1], cmap='Greens')
        axarr[0, 2].imshow(img[vmin:vmax, umin:umax, 2], cmap='Blues')
    axarr[1, 0].matshow(features[vmin:vmax, umin:umax, 0])
    axarr[1, 1].matshow(features[vmin:vmax, umin:umax, 1])
    axarr[1, 2].matshow(features[vmin:vmax, umin:umax, 2])
    axarr[2, 0].matshow(features[vmin:vmax, umin:umax, 3])
    axarr[2, 1].matshow(features[vmin:vmax, umin:umax, 4])
    axarr[2, 2].matshow(features[vmin:vmax, umin:umax, 5])
    axarr[0, 3].matshow(features[vmin:vmax, umin:umax, 6])
    axarr[1, 3].matshow(features[vmin:vmax, umin:umax, 7])
    axarr[2, 3].matshow(features[vmin:vmax, umin:umax, 8])
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', type=str, default='/media/huifang/data/registration/mouse/anterior_v1/', help='rootpath to data')
opt = parser.parse_args()

#read files
img = opt.rootpath + '/spatial/tissue_lowres_image.png'
filtered_h5 = opt.rootpath + 'filtered_matrix.h5'
spatial_file = opt.rootpath + 'spatial/tissue_positions_list.csv'
if not os.path.exists(spatial_file):
    spatial_file = opt.rootpath + 'spatial/tissue_positions_list.txt'
scale_file = opt.rootpath + 'spatial/scalefactors_json.json'
low_res_scale = get_res_scale(scale_file, 'tissue_lowres_scalef')

dense_threshold = 0.9
spot_x, spot_y, pixel_u, pixel_v = get_indices_from_file(spatial_file,low_res_scale)
spot_y, spot_x = regularize_spot_scale(pixel_v,pixel_u,spot_y,spot_x)
filtered_matrix_h5 = get_matrix_from_h5(filtered_h5,dense_threshold)
gene_feature_dimension = filtered_matrix_h5.shape[0]
print("Reduced feature dimension is ", gene_feature_dimension,", under a threshold of ", dense_threshold)

img = get_images_from_file(img)
img_height = img.shape[0]
img_width = img.shape[1]

featuremaps = np.zeros([img_height,img_width,gene_feature_dimension])

featuremaps = filling_maps(featuremaps,filtered_matrix_h5,pixel_v,pixel_u,img_height,img_width)
spot_height = np.amax(spot_y)+1
spot_width = np.amax(spot_x)+1
countmaps = np.zeros([spot_height,spot_width,gene_feature_dimension])
countmaps = filling_maps(countmaps,filtered_matrix_h5,spot_y,spot_x,spot_height,spot_width)


showfeaturechanels(featuremaps,pixel_v,pixel_u,img)
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


