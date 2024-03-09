import math
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
from neurora.nii_save import corr_save_nii


def isc_roi(data, roi_mask):
    '''
    NeuroRA asks for a data shape of [time_points, subjects, x, y, z] to calculate
    correlation on the spatial signal among subjects, returning [time_points, matrix, 2].
    For rsfMRI, I want to calculate the correlation on the time dimension among subjects,
    so I modify their code
    input shape = [subjects, x, y, z, time_points], output shape = [half matrix(subj x subj), 2]

    :param data: fMRI data of all subjects [subjects, x, y, z, time_points].
                 At least 2 subjects are needed
    :param roi_mask: Resolution of the ROI should be the same as the fMRI data (in spatial dimensions)
    :return: ISC_results
    '''
    data = data.transpose([0, 4, 1, 2, 3])
    # print('combine shape=', data.shape)  # [subjects, time_point, x, y, z]
    # print('roi shape=', roi_mask.shape)
    # the number of pairs among n_subs
    nsubs, nts, nx, ny, nz = data.shape
    if nsubs > 2:
        n = int(math.factorial(nsubs) / (2 * math.factorial(nsubs - 2)))
    elif nsubs == 2:
        n = 1
    else:
        print('At least 2 subjects are needed!')
        exit()

    # average spatial signals in each time point
    temp_data = np.full([nsubs, nts], np.nan)
    data = np.multiply(data, roi_mask)  # mask must be binarized to 0 1
    valid_data = (roi_mask != 0)
    data_in_valid = np.sum(valid_data)
    for sub in range(nsubs):
        for t in range(nts):
            data_sum = np.sum(data[sub, t])
            temp_data[sub, t] = data_sum / data_in_valid

    # calculate ISC, using pearson, keep the correlation direction (+/-)
    isc_result = np.full([n, 2], np.nan)
    # print('Calculating ISC...')
    this_n = 0
    for sub_i in range(nsubs):
        for sub_j in range(nsubs):
            if sub_i < sub_j:
                if (np.isnan(temp_data[sub_i]).any() == False) and (np.isnan(temp_data[sub_j]).any() == False):
                    isc_result[this_n] = pearsonr(temp_data[sub_i], temp_data[sub_j])
                this_n += 1
    return isc_result


def isc_searchlight(data):
    '''
    :param data: shape as [subjects, x,y,z, TP]
    :return: ISC matrix, RDM, both of [nx, ny, nz, n], not save p-value
    '''
    nsubs, nx, ny, nz, nts = data.shape

    if nsubs > 2:
        n = int(math.factorial(nsubs) / (2 * math.factorial(nsubs - 2)))
    elif nsubs == 2:
        n = 1
    else:
        print('At least 2 subjects are needed!')
        exit()

    # initial output data
    corrs = np.full([nx, ny, nz, n], np.nan)  # store RSA results, r- and p-value in each voxel
    fmri_rdm = np.full([nx, ny, nz, n], np.nan)
    # print(f'Calculating ISC...')
    # calculate ISC in each voxel (across TP)
    for this_x in range(nx):
        for this_y in range(ny):
            for this_z in range(nz):
                this_n = 0
                for sub_i in range(nsubs):
                    for sub_j in range(nsubs):
                        if sub_i < sub_j:
                            if (np.isnan(data[sub_i, this_x, this_y, this_z]).any() == False) and (
                                    np.isnan(data[sub_j, this_x, this_y, this_z]).any() == False):
                                r, p = pearsonr(data[sub_i, this_x, this_y, this_z],
                                                data[sub_j, this_x, this_y, this_z])
                                corrs[this_x, this_y, this_z, this_n] = r
                                fmri_rdm[this_x, this_y, this_z, this_n] = 1 - r
                            this_n += 1
    return corrs, fmri_rdm


def cal_behav_rdm(labels):
    matrix_n = int(math.factorial(len(labels)) / (2 * math.factorial(len(labels) - 2)))
    distance_RDM = np.full(matrix_n, np.nan)
    n_index = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i < j:
                distance_RDM[n_index] = abs(labels[i] - labels[j])
                n_index += 1
    return distance_RDM


def process_isc_roi(q, fmri_list, roi_mask, i, j):
    '''pair isc, for multiple processing'''
    img1 = nib.load(fmri_list[i]).get_fdata()
    img2 = nib.load(fmri_list[j]).get_fdata()
    this_pair = np.array([img1, img2])
    this_isc_result = isc_roi(this_pair, roi_mask)
    q.put([this_isc_result[0][0], [i, j]])


def process_isc_searchlight(q, fmri_list, i, j, index):
    '''pair isc, for multiple processing'''
    img1 = nib.load(fmri_list[i]).get_fdata()
    img2 = nib.load(fmri_list[j]).get_fdata()
    this_pair = np.array([img1, img2])
    corrs, fmri_rdm = isc_searchlight(this_pair)
    q.put([corrs, fmri_rdm, [i, j], index])


def process_isc_rsa(label_file, this_temp, nx, ny, nz, matrix_n, result_dir, affine):
    labels = label_file[this_temp]
    distance_RDM = cal_behav_rdm(labels)
    rsa_result = np.full([nx, ny, nz, 2], np.nan)
    print('Generating fmri_rdm...')
    for this_x in range(nx):
        for this_y in range(ny):
            for this_z in range(nz):
                fmri_rdm = np.full(matrix_n, np.nan)
                for this_n in range(matrix_n):
                    fmri_rdm[this_n] = np.load(fr'{result_dir}/temp/rdm_result_{this_n}.npy')[this_x, this_y, this_z]
                rsa_result[this_x, this_y, this_z] = np.array(spearmanr(fmri_rdm, distance_RDM))
    print('Generation done, making image...')
    np.save(rf'{result_dir}/{this_temp}_search_rsa', rsa_result)
    print('saved RSA results as {result_dir}/{this_temp}_search_rsa.npy')
    print('generating RSA result image...')
    img = corr_save_nii(rsa_result, affine, rf'results/{this_temp}_search_rsa_img.nii', size=[nx, ny, nz],
                        ksize=[1, 1, 1], r=-1, plotrlt=False)


def process_rsa_element(q, this_voxel, matrix_n, result_dir, distance_RDM):
    fmri_rdm = np.full(matrix_n, np.nan)
    for this_n in range(matrix_n):
        fmri_rdm[this_n] = np.load(fr'{result_dir}/temp/isc_result_{this_n}.npy')[
            this_voxel[0], this_voxel[1], this_voxel[2]]
    this_result = np.array(spearmanr(fmri_rdm, distance_RDM))
    q.put([this_voxel, this_result])


def multi_save_fmri_rdm(q, matrix_n, result_dir, this_ind):
    fmri_rdm = np.full(matrix_n, np.nan)
    for this_n in range(matrix_n):
        fmri_rdm[this_n] = np.load(fr'{result_dir}/temp/rdm_result_{this_n}.npy')[this_ind[0], this_ind[1], this_ind[2]]
    np.save(rf'{result_dir}/temp/fmri_rdm_{this_ind[0]}_{this_ind[1]}_{this_ind[2]}', fmri_rdm)
    q.put(1)
