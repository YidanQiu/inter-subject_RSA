import glob
import math
import multiprocessing
import nibabel as nib
import numpy as np
import pandas as pd
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.stuff import get_affine, permutation_corr
from neurora.nii_save import corr_save_nii
from scipy.stats import pearsonr, spearmanr
from my_ISC_functions import process_isc_searchlight, process_isc_rsa, cal_behav_rdm, multi_save_fmri_rdm

result_dir = 'results'  # name of the directory to store results
rdm_dir = r'rsa_result'
label_file = pd.read_excel('subj_info.xlsx')  # read excel that stores behavioral dada of subjects
behav = 'depressive'  # variable name

# load fmri data; shape=[subjects, x, y, z, time_points]; calculate ISC element by element
fmri_dir = 'subj_data/*.nii'
fmri_list = glob.glob(fmri_dir)
fmri_list.sort()

affine = get_affine(fmri_list[0])
nx, ny, nz, nt = nib.load(fmri_list[0]).get_fdata().shape
num_subj = len(label_file['sub'])
matrix_n = int(math.factorial(num_subj) / (2 * math.factorial(num_subj - 2)))

# Generate fmri RDM
voxel_index = []
for this_x in range(nx):
    for this_y in range(ny):
        for this_z in range(nz):
            voxel_index.append([this_x, this_y, this_z])


this_label = label_file[behav]
behav_RDM = cal_behav_rdm(this_label)

print(behav)
rsa_result = np.full([nx, ny, nz, 2], np.nan)
for this_voxel in voxel_index:
    fmri_rdm = np.load(rf'{rdm_dir}/fmri_rdm_{this_voxel[0]}_{this_voxel[1]}_{this_voxel[2]}.npy')
    this_result = np.array(spearmanr(fmri_rdm, behav_RDM))
    this_result[1] = permutation_corr(fmri_rdm, behav_RDM, method="spearman", iter=10000)
    rsa_result[this_voxel[0], this_voxel[1], this_voxel[2]] = this_result
np.save(rf'{result_dir}/{behav}_search_rsa', rsa_result)
print(f'saved RSA results as {result_dir}/{behav}_search_rsa.npy')
print('generating RSA result image...')
img = corr_save_nii(rsa_result, affine, rf'results/{behav}_search_rsa_img_p05.nii', size=[nx, ny, nz],
                    ksize=[1, 1, 1], r=-1, p=0.01, correct_method='Cluster-FWE', clusterp=0.05, plotrlt=False)
