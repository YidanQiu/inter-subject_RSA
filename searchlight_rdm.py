import glob
import os
import math
import multiprocessing
import nibabel as nib
import numpy as np
import pandas as pd
from neurora.stuff import get_affine
from neurora.nii_save import corr_save_nii
from scipy.stats import pearsonr, spearmanr
from my_ISC_functions import process_isc_searchlight, process_isc_rsa, cal_behav_rdm, multi_save_fmri_rdm

rdm_dir = 'results'  # name of the directory storing RDMs
result_dir = r'rsa_result'  # name of the directory to store results

# load fmri data; shape=[subjects, x, y, z, time_points]; calculate ISC element by element
fmri_dir = 'subj_data/*.nii'
fmri_list = glob.glob(fmri_dir)
fmri_list.sort()

affine = get_affine(fmri_list[0])
nx, ny, nz, nt = nib.load(fmri_list[0]).get_fdata().shape
num_subj = len(fmri_list)
matrix_n = int(math.factorial(num_subj) / (2 * math.factorial(num_subj - 2)))
matrix_ind = np.arange(0, matrix_n, 1)

# Generate fmri RDM
print('Generating fmri_rdm...')
voxel_index = []
for this_x in range(nx):
    for this_y in range(ny):
        for this_z in range(nz):
            voxel_index.append([this_x, this_y, this_z])

num_ok = 0
print(matrix_n)

for this_ind in voxel_index:
    this_ok = 0
    fmri_rdm = np.full(matrix_n, np.nan)
    for this_n in range(matrix_n):
        fmri_rdm[this_n] = np.load(fr'{rdm_dir}/rdm_result_{matrix_ind[this_n]}.npy')[
            this_ind[0], this_ind[1], this_ind[2]]
        this_ok += 1
        print(f'\rDone {this_ok}, {round(num_ok / matrix_n * 100, 2)}, progress: {this_ok / matrix_n * 100}',
              end='')
    np.save(rf'{result_dir}/fmri_rdm_{this_ind[0]}_{this_ind[1]}_{this_ind[2]}', fmri_rdm)
    num_ok += 1
