import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import paste as pst
import scanpy as sc
import anndata
from sklearn.decomposition import NMF
import os
style.use('seaborn-dark')
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

path_to_output_dir = '../data/SCC/cached-results/'


def pairwise_exp(adata1, adata2):
    # Run pairwise align
    pi = pst.pairwise_align(adata1, adata2, alpha=0.1)

    spotsA, spotsB = np.nonzero(pi)
    s = 0
    for i in range(len(spotsA)):
        # get the clusters corresponding to each spot
        a = adata1.obs['original_clusters'][spotsA[i]]
        b = adata2.obs['original_clusters'][spotsB[i]]
        if a == b:
            s += pi[spotsA[i]][spotsB[i]]
    return s

path_to_h5ads = path_to_output_dir + 'H5ADs/'

patient_2 = []
patient_5 = []
patient_9 = []
patient_10 = []

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}

for k in patients.keys():
    for i in range(3):
        patients[k].append(sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad'))




df = pd.DataFrame()
for k, patient_n in patients.items():
    a = pairwise_exp(patient_n[0].copy(), patient_n[1].copy())
    b = pairwise_exp(patient_n[0].copy(), patient_n[2].copy())
    df[k] = [a, b]

results_dir = path_to_output_dir + 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
df.to_csv(results_dir + 'center_results.csv')

center_slice, pis = pst.center_align(initial_slice, patient_n, lmbda, random_seed = 5)