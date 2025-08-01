import numpy as np
from brainspace.gradient import GradientMaps

def normalize_list(data_list):
    """
    주어진 리스트의 값을 0과 1 사이로 정규화합니다 (Min-Max Normalization).

    Args:
        data_list: 정규화할 숫자 리스트.

    Returns:
        정규화된 값을 포함하는 새 리스트.
        입력 리스트가 비어 있거나 모든 값이 동일하면 빈 리스트 또는 0으로 채워진 리스트를 반환합니다.
    """
    if not data_list:
        return []

    
    min_val = min(data_list)
    max_val = max(data_list)

    # 모든 값이 동일한 경우 (분모가 0이 되는 것을 방지)
    if max_val == min_val:
        return [0.0] * len(data_list)

    normalized_list = []
    
    sum=0;
    for x in data_list:
        sum+=x*x;
    
    for y in data_list:
        if sum>0:
            normalized_y = y/np.sqrt(sum);
        else:
            normalized_y=0;
        #normalized_list.append(normalized_y);
    
        
    for x in data_list:
        normalized_x = (x - min_val) / (max_val - min_val)
        normalized_list.append(normalized_x)
        
    
        
    return normalized_list

def fun_fg(fc_mat):
    gm = GradientMaps(n_components=2, approach='dm', kernel='cosine')
        
    gradients_data1=[];
    gradients_data2=[];

    ref_grad = None
        
    for i in range(len(fc_mat)):
            
        gm.fit(fc_mat[i]);
        gradients = gm.gradients_       # shape: (n_regions, 5)
        gradients[:,0]=normalize_list(gradients[:,0].tolist())
        gradients[:,1]=normalize_list(gradients[:,1].tolist())
            
        gradients_data1.append(  gradients[:,0]);
        gradients_data2.append(  gradients[:,1]);

    return np.array(gradients_data1), np.array(gradients_data2);


def fun_fc_vector(smoothed_data, window_size,step_size):
    
    n_neurons=len(smoothed_data[:,1]);
    
    n_timebins=len(smoothed_data[1,:]);
    
    #print(n_neurons,n_timebins)
    
    n_windows = (n_timebins - window_size) // step_size + 1

    fc_mat=[];#np.zeros((n_neurons,n_neurons));
    # -------------------------------
    # 3. 각 윈도우마다 correlation matrix 계산
    # -------------------------------
    # 각 윈도우에서 상삼각 요소들을 저장할 리스트
    fc_vectors = []
    ave_fc_mat=np.zeros((n_neurons,n_neurons));
    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        window_data = smoothed_data[:, start:end]  # shape: (neurons, time)
        
        # 피어슨 상관계수 행렬 계산
        corr_matrix = np.corrcoef(window_data)
        np.fill_diagonal(corr_matrix, 0)
        fc_mat.append(corr_matrix);
        
        ave_fc_mat+=corr_matrix 
        # NaN (분산 0) 제거
        #corr_matrix = np.nan_to_num(corr_matrix)

        # 상삼각 요소 추출 (i < j)
        upper_tri = corr_matrix[np.triu_indices(n_neurons, k=1)]
        fc_vectors.append(upper_tri)

    # 결과 shape: (n_windows, num_connections)
    fc_vectors = np.array(fc_vectors) 

    return fc_vectors, fc_mat,ave_fc_mat/n_windows;



def generate_sc_weight(sc_mat, CellXYZ_data, log_std):
    np.random.seed(42) 
    
    #file_name1=f'C:/Users/044ap/OneDrive/2025_zebrafish_topology/heavy_tailed_topology_function/raw_data/subject12_cellular_data.npz';
    #file_name2=f'C:/Users/044ap/OneDrive/2025_zebrafish_topology/heavy_tailed_topology_function/raw_data/subject12_cellular_data_sc_mat_indices.pkl';

    #save_file_name=f'./subject_12_base_sc_for_simualtion.pkl';

    #data = np.load(file_name1, allow_pickle=True)
    #sel_neuron = np.array(data['sel_neuron']);
    #sc_mat=np.array(data['sc_mat']);

    #file_name1=f'C:/Users/044ap/OneDrive/2025_zebrafish_topology/heavy_tailed_topology_function/raw_data/subjet_12_inter_area_strengthen_false_sc_mat.npy';
    #sc_mat =np.load(file_name1);
    #save_file_name=f'./subject_12_synaptic_weight_high_sc_for_simualtion.pkl'

    #CellXYZ_data = np.array(data['t_CellXYZ_data'][sel_neuron]);
    #CellXYZ_data=CellXYZ_data.astype(float);
    #t_spot_data = data['t_spot_data']


    #with open(file_name2, "rb") as f:
    #    indices_ii = pickle.load(f)
    
    num_neurons = len(CellXYZ_data);
    print(f"Neuron 수: {num_neurons}")

    # 상삼각 인덱스 추출
    i_indices, j_indices = np.triu_indices(num_neurons, k=1)
    mask = sc_mat[i_indices, j_indices] > 0
    i_conn = i_indices[mask]
    j_conn = j_indices[mask]

    # 거리 계산 (벡터 연산)
    coords_i = CellXYZ_data[i_conn]
    coords_j = CellXYZ_data[j_conn]

    diff=coords_i-coords_j;
    dists = np.linalg.norm(diff, axis=1)
 

    # 무작위 방향 설정
    rand_mask = np.random.rand(len(i_conn)) > 0.5

    src = np.where(rand_mask, i_conn, j_conn)
    dst = np.where(rand_mask, j_conn, i_conn)

    custom_connections = [[] for _ in range(num_neurons)];
    connection_dist=[[] for _ in range(num_neurons)];
    weight_dist=[[] for _ in range(num_neurons)];

    log_mean=1.23*np.exp(-0.05*dists)-5.5611;
        #log_std=1.0; #0.01,0.1,1.0;
    weights = np.random.lognormal(mean=log_mean, sigma=log_std)
    
    for tsrc, tdgt, tdist,weight in zip(src, dst, dists, weights):
        custom_connections[tsrc].append(int(tdgt.tolist()))
        connection_dist[tsrc].append(int((tdist.tolist())))
        
        #log_mean=1.23*np.exp(-0.05*tdist)-5.5611;
        #log_std=1.0; #0.01,0.1,1.0;
        #weight = np.random.lognormal(mean=log_mean, sigma=log_std);
        #weight=1.5*np.exp(-tdist/30);
        weight_dist[tsrc].append(weight);


    #save_file_name=f'./subject_12_base_sc_for_simualtion_low_variance.pkl';

    data_to_save = {
        'custom_connections': custom_connections,
        'connection_dist': connection_dist,
        'weight_dist': weight_dist
    }

    return data_to_save;
                

