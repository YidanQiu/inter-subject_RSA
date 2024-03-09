import glob
import math
import multiprocessing
import nibabel as nib
import numpy as np
import pandas as pd
from neurora.stuff import get_affine
from neurora.nii_save import corr_save_nii
from scipy.stats import pearsonr, spearmanr
from my_ISC_functions import process_isc_searchlight, process_isc_rsa, cal_behav_rdm

result_dir = 'results'  # name of the directory to store results
label_file = pd.read_excel('subj_info.xlsx')  # read excel that stores behavioral dada of subjects
temps = ['depressive', 'cyclothymic', 'hyperthymic', 'irritable', 'anxious', 'sum']  # column name of each variable

# load fmri data; shape=[subjects, x, y, z, time_points]; calculate ISC element by element
fmri_dir = 'subj_data/*.nii'
fmri_list = glob.glob(fmri_dir)
fmri_list.sort()

affine = get_affine(fmri_list[0])
nx, ny, nz, nt = nib.load(fmri_list[0]).get_fdata().shape
num_subj = len(fmri_list)
matrix_n = int(math.factorial(num_subj) / (2 * math.factorial(num_subj - 2)))
print(f'got data from {num_subj} subjects.')
print(f'data shape as [{nx},{ny},{nz}], with {nt} time points.')

subj_ind = []
for i in range(num_subj):
    for j in range(num_subj):
        if i < j:
            subj_ind.append([i, j])

print(f'calculating {matrix_n} correlations...')

# multiple processes
q = multiprocessing.Manager().Queue()
po = multiprocessing.Pool(10)  # number of processes
for pair in subj_ind:
    po.apply_async(process_isc_searchlight,
                   args=(q, fmri_list, pair[0], pair[1]))
po.close()
num_ok = 0
while True:
    single_isc_result = q.get()  # [corrs, fmri_rdm, [i, j]]
    this_ind = subj_ind.index(single_isc_result[2])
    this_isc_result = single_isc_result[0].squeeze(-1)
    this_rdm_result = single_isc_result[1].squeeze(-1)
    np.save(fr'{result_dir}/isc_result_{this_ind}', this_isc_result)
    np.save(fr'{result_dir}/rdm_result_{this_ind}', this_rdm_result)
    num_ok += 1
    print('\rprogress: %.2f%%' % (num_ok / matrix_n * 100), end='')
    if num_ok >= matrix_n:
        break

