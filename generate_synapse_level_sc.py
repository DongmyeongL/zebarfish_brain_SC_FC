
import numpy as np
import os
import scipy.spatial.distance as ssd
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import scipy.io
import time
import analysis_assitance_python as aap
import time 
import random
import seaborn as sns

def load_mat_data(path, variable_name):
    """ .mat 파일에서 특정 변수를 로드합니다. """
    try:
        data = scipy.io.loadmat(path)
        if variable_name not in data:
            raise ValueError(f"'{variable_name}' 변수가 {path} 파일에 없습니다.")
        return data[variable_name]
    except FileNotFoundError:
        print(f"오류: {path} 파일을 찾을 수 없습니다.")
        raise
    except Exception as e:
        print(f"{path} 로딩 중 오류 발생: {e}")
        raise
def sort_clusters_by_spatial_criteria(means):
    """
    Sorts cluster indices based on spatial coordinates (x then y).
    Clusters with y <= 300 are considered 'right' and sorted first,
    followed by 'left' clusters (y > 300).

    Args:
        means (np.ndarray): A 2D array where each row is the [x, y, z] mean
                            coordinate of a cluster.

    Returns:
        np.ndarray: An array of sorted cluster indices.
    """
    idx_sorted_x = np.argsort(means[:, 0])
    y_sorted = means[idx_sorted_x, 1]
    right_indices = idx_sorted_x[y_sorted <= 300]
    left_indices = idx_sorted_x[y_sorted > 300]
    return np.concatenate((right_indices, left_indices))
   
    
numRegions = 72

# File Paths
NEURON_ENDPOINTS_PATH = '../raw_data/neuronEndpoints_data.mat'

neuronEndpoints = load_mat_data(NEURON_ENDPOINTS_PATH, 'neuronEndpoints')

# Process neuronEndpoints
if neuronEndpoints.ndim == 2 and 1 in neuronEndpoints.shape:
    items = [arr.squeeze() if isinstance(arr, np.ndarray) else np.array([]) for arr in neuronEndpoints.flatten()]
else:
    items = [arr if isinstance(arr, np.ndarray) else np.array([]) for arr in neuronEndpoints]
neuronEndpoints = items

# --- Synapse Coordinate Transformation ---
a, b = 1.18, 54.42795
c, d = 1.204, -32.267997
e, f = 0.374574, 9.218573

all_syn = []
syn_idx = []

for i, ep_raw in enumerate(neuronEndpoints):
    ep = np.array(ep_raw)
    if ep.size == 0:
        continue

    if ep.ndim == 1 and ep.size == 4: # Handle single 4-element arrays (assuming [y, x, z, other_val])
        x = ep[1] * a + b
        y = ep[0] * c + d
        z = ep[2] * e + f
        syn_xyz = np.array([x, y, z])
        all_syn.append(syn_xyz)
        syn_idx.append(i) # Single index for a single point
    elif ep.ndim == 2: # Handle 2D arrays (multiple [y, x, z, other_val] rows)
        x = ep[:, 1] * a + b
        y = ep[:, 0] * c + d
        z = ep[:, 2] * e + f
        syn_xyz = np.stack([x, y, z], axis=1)
        all_syn.append(syn_xyz)
        syn_idx.extend(np.full(len(syn_xyz), i).tolist()) # Extend with multiple indices
    else:
        print(f"Warning: Unexpected shape for neuronEndpoints[{i}]: {ep.shape}. Skipping.")

syn_xyz = np.vstack(all_syn).astype(np.float64)
syn_idx = np.array(syn_idx, dtype=np.int32)


FILE_PATHS = [
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_12_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_13_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_14_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_15_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_16_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_17_data.mat',
    r'D:\one_drive\OneDrive\zebra_fish_connection_large_scaling_modeling\zebra_fish_calcuim_imaging\zebra_fish_calcuim_imaging\raw_calcuim_imaging_data\subject_18_data.mat',
]


save_file_path_synapse_sc = [
    f'../raw_data/subject_12_data_synapse_sc_mat.py',
    f'../raw_data/subject_13_data_synapse_sc_mat.py',
    f'../raw_data/subject_14_data_synapse_sc_mat.py',
    f'../raw_data/subject_15_data_synapse_sc_mat.py',
    f'../raw_data/subject_16_data_synapse_sc_mat.py',
    f'../raw_data/subject_17_data_synapse_sc_mat.py',
    f'../raw_data/subject_18_data_synapse_sc_mat.py'
]

save_file_patha_synapse_sc_aux_data = [
    f'../raw_data/subject_12_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_13_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_14_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_15_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_16_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_17_data_synapse_sc_aux_data.pkl',
    f'../raw_data/subject_18_data_synapse_sc_aux_data.pkl',
]


numSubjects = len(FILE_PATHS)

NUM_REGIONS=72;
for file_idx in range(numSubjects):
    print(f"Processing subject {file_idx + 1}/{numSubjects}...")

    try:
        CellXYZ_full = load_mat_data(FILE_PATHS[file_idx], 'CellXYZ')
        result_idx_mat = load_mat_data(FILE_PATHS[file_idx], 'result_idx')
        spot_data_full = load_mat_data(FILE_PATHS[file_idx], 'spot_data')
    except Exception as e:
        print(f"Error loading data for subject {file_idx + 1}: {e}. Skipping this subject.")
        continue

    # Ensure result_idx_cells is always 1D
    if result_idx_mat.ndim > 1:
        result_idx_cells = result_idx_mat.flatten()
    else:
        result_idx_cells = result_idx_mat

   
    ave_xyz=[];
    for r_idx in range(NUM_REGIONS):
        
        ids_1_based_in_cell = result_idx_cells[r_idx]

            # Handle cases where ids_1_based_in_cell might be a scalar or a single-element array
        if np.isscalar(ids_1_based_in_cell):
                ids_1_based = np.array([ids_1_based_in_cell])
        else:
                ids_1_based = ids_1_based_in_cell.flatten().astype(int)

        ids_0_based = ids_1_based - 1 # important
        ave_xyz.append(np.mean(CellXYZ_full[ids_0_based, :],0));
       
        
    sort_id=sort_clusters_by_spatial_criteria(np.array(ave_xyz)); 
    
    #averaging 
    ave_region_mean=[];
    
    s_id=[]; 
    for r_idx in sort_id:
        
        ids_1_based_in_cell = result_idx_cells[r_idx]

            # Handle cases where ids_1_based_in_cell might be a scalar or a single-element array
        if np.isscalar(ids_1_based_in_cell):
                ids_1_based = np.array([ids_1_based_in_cell])
        else:
                ids_1_based = ids_1_based_in_cell.flatten().astype(int)

        ids_0_based = ids_1_based - 1
        if(len(ids_0_based)>0):
            s_id.append(r_idx);
            ave_region_mean.append(np.mean(spot_data_full[ids_0_based,:],0));
    
        
    t_num_region=[];
    region_num=[];
    check_id=[];
    for r_idx in sort_id:
        
        ids_1_based_in_cell = result_idx_cells[r_idx]

            # Handle cases where ids_1_based_in_cell might be a scalar or a single-element array
        if np.isscalar(ids_1_based_in_cell):
                ids_1_based = np.array([ids_1_based_in_cell])
        else:
                ids_1_based = ids_1_based_in_cell.flatten().astype(int)

        ids_0_based = ids_1_based - 1
        random.shuffle(ids_0_based)    
        t_num_region.extend(ids_0_based);
        
        if(len(ids_0_based)):
            region_num.append(len(ids_0_based))
            for k in range(len(ids_0_based)):
                check_id.append(r_idx);
        
    region_interval=[];
    imax=0;
    for i in range(len(region_num)):
        region_interval.append([imax,imax+region_num[i]])
        imax+=region_num[i];
        
    t_CellXYZ_data = CellXYZ_full[t_num_region, :]
    t_spot_data = spot_data_full[t_num_region, :]
        
    print('Calculating sc from synapse......')
    sel_neuron,sc_mat=aap.calculate_sc_from_synapse(t_CellXYZ_data.tolist(),syn_xyz.tolist(),syn_idx.tolist());
    
    t_CellXYZ_data= t_CellXYZ_data[sel_neuron,:];
    t_spot_data=t_spot_data[sel_neuron,:]; 
    
    s_check_id=[];
    check_id=np.array(check_id);
    
    for i in range(len(sel_neuron)):
        
        t_id=check_id[sel_neuron[i]];
        s_check_id.append(t_id);
        
    ts_check_id=list(dict.fromkeys(s_check_id));
    
    def find_all_indices(input_list, target_element):
        """
        리스트에서 특정 원소가 나타나는 모든 인덱스를 찾아 리스트로 반환합니다.
        """
        indices = []
        index=0;
        for element in (input_list):
            if element == target_element:
                indices.append(index)
            index+=1;
            
        return indices;
    
        
    indices_ii=[];
    indice_interval=[];
    
    synapse_id=[];
    
    
    for i in range(len(sort_id)):
        ii=sort_id[i];
        temp =find_all_indices(s_check_id, ii)
        
        if(len(temp)>0):
            indices_ii.append(temp);
            synapse_id.append(ii)
            
        if(len(temp)>0):
            minx = np.min(temp)
            maxx = np.max(temp)
            indice_interval.append([minx,maxx]);

    
    synapse_region_mean=[];
    for ii in range(len(synapse_id)):
        synapse_region_mean.append(np.mean(t_spot_data[indices_ii[ii],:],0));
    
    n = len(indices_ii)
    re_sc_mat = np.zeros((n, n))
    
    sc_mat=np.array(sc_mat);
    for i in range(n):
        for j in range(i + 1, n):
            
            neurons_i = indices_ii[i]
            neurons_j = indices_ii[j]
            t_sum = sc_mat[np.ix_(neurons_i, neurons_j)].sum();
            re_sc_mat[i, j] = t_sum
            re_sc_mat[j, i] = t_sum
            
            #pairs = [(a, b) for a in indices_ii[i] for b in indices_ii[j]]
            #if pairs:
            #    values = [sc_mat[a][b] for a, b in pairs]
            #    re_sc_mat[i, j] = re_sc_mat[j, i] = sum(values)

    np.save(save_file_path_synapse_sc[file_idx],sc_mat);
    
    data_to_save = {
    're_sc_mat': re_sc_mat,
    'synapse_region_mean': synapse_region_mean,
    't_CellXYZ_full': t_CellXYZ_data,
    'syn_xyz': syn_xyz,
    'syn_idx': syn_idx,
    't_spot_data': t_spot_data,
    'synapse_id': synapse_id,
    'indices_ii': indices_ii
    }
    import pickle
    with open(save_file_patha_synapse_sc_aux_data[file_idx], 'wb') as f:
        pickle.dump(data_to_save, f)
    
    
    
    
    '''
    plt.figure();
    plt.imshow(np.log10(re_sc_mat+1));
        
    plt.figure();
    plt.imshow(np.corrcoef(synapse_region_mean))
        
    plt.figure();
    plt.imshow(np.corrcoef(ave_region_mean))
    
    plt.figure();
    plt.scatter(CellXYZ_full[:,0],CellXYZ_full[:,1],s=0.1,alpha=0.3);
    plt.scatter(syn_xyz[:,0],syn_xyz[:,1],s=0.1,alpha=0.3);
    plt.show();
    '''

    
    
    