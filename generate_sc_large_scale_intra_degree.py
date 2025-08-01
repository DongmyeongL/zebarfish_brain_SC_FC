import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from random import uniform
from colorsys import hsv_to_rgb
from scipy.stats import linregress
from scipy.stats import lognorm, gaussian_kde, norm, kurtosis, skew, expon
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import powerlaw 
from scipy.stats import lognorm, gamma
from scipy.optimize import curve_fit
import networkx as nx

def detect_core_nodes(SC, core_ratio=0.25):
    """
    Degree 기반 Core-Periphery 구분
    """
    degrees = SC.sum(axis=0)
    N = len(degrees)
    num_core = max(1, int(N * core_ratio))
    core_indices = np.argsort(degrees)[-num_core:]  # degree 높은 순
    return list(core_indices)

def modify_core_periphery(SC, core_nodes, strengthen=True, seed=42):
    """
    Core-Periphery 구조 강화 또는 약화
    """
    np.random.seed(seed)
    random.seed(seed)

    N = SC.shape[0]
    upper_SC = np.triu(SC, 1)
    edges = np.transpose(np.nonzero(upper_SC))
    num_edges = len(edges)

    core_set = set(core_nodes)
    periphery_set = set(range(N)) - core_set

    core_core = []
    core_periph = []
    periph_periph = []

    # 분류
    for i, j in edges:
        if i in core_set and j in core_set:
            core_core.append((i, j))
        elif (i in core_set and j in periphery_set) or (j in core_set and i in periphery_set):
            core_periph.append((i, j))
        else:
            periph_periph.append((i, j))

    if strengthen and len(periph_periph) > 10:
        # Periphery 간 연결 일부 제거 → Core-Periphery로 전환
        n_swap = int(len(periph_periph) * 0.6)
        for _ in range(n_swap):
            i, j = periph_periph.pop(random.randrange(len(periph_periph)))
            t = i if np.random.rand() < 0.5 else j
            k = random.choice(list(core_set))
            core_periph.append((t, k))

    elif not strengthen and len(core_core) > 10:
        # Core 간 연결 일부 제거 → Core-Periphery 또는 Periphery 간으로 전환
        n_swap = int(len(core_core) * 0.8)
        for _ in range(n_swap):
            i = random.choice(list(periphery_set))
            if np.random.rand() < 0.5 and core_core:
                a, b = core_core.pop(random.randrange(len(core_core)))
                j = a if np.random.rand() < 0.5 else b
                core_periph.append((i, j))
            elif core_periph:
                a, b = core_periph.pop(random.randrange(len(core_periph)))
                periph_periph.append((i, a))

    # 새 SC 행렬 생성
    new_edges = core_core + core_periph + periph_periph
    new_SC = np.zeros_like(SC)
    for i, j in new_edges:
        new_SC[i, j] = 1
        new_SC[j, i] = 1

    return new_SC

data = np.load('../raw_data/subject12_cellular_data.npz', allow_pickle=True)
sel_neuron = data['sel_neuron']
sc_mat = data['sc_mat']
t_CellXYZ_data = data['t_CellXYZ_data']
t_spot_data = data['t_spot_data']


with open("../raw_data/subject12_cellular_data_sc_mat_indices.pkl", "rb") as f:
    indices_ii = pickle.load(f)
    
total_new_id=[];
for i in range(len(indices_ii)):
    pick_num_i = indices_ii[i]
    sub_sc_i = sc_mat[np.ix_(pick_num_i, pick_num_i)]
    new_sort_id=np.argsort(np.sum(sub_sc_i,axis=0));
    total_new_id.extend(new_sort_id+ pick_num_i[0]);

#new_id=np.array(total_new_id);
#sc_mat = sc_mat[new_id][:, new_id];

#degree distribution rewring each area    
plt.figure();
for i in range(len(indices_ii)):
    pick_num_i = indices_ii[i]
    sub_sc_i = sc_mat[np.ix_(pick_num_i, pick_num_i)]

    #new_sort_id=np.argsort(np.sum(sub_sc_i,axis=0));
    #sub_sc_i = sub_sc_i[ new_sort_id][:,  new_sort_id];
    
    core_nodes = detect_core_nodes(sub_sc_i, core_ratio=0.3);
    new_sub_sc_i = modify_core_periphery(sub_sc_i, core_nodes, strengthen=False);    
    
    sc_mat[np.ix_(pick_num_i, pick_num_i)]=new_sub_sc_i;
    
    #plt.subplot(1,2,1);plt.imshow(sub_sc_i);
    #plt.subplot(1,2,2);plt.imshow(new_sub_sc_i);
    #plt.show();
    
    #new_sort_id=np.argsort(np.sum(sub_sc_i,axis=0));
    #total_new_id.extend(new_sort_id+ pick_num_i[0]);

#new_id=np.array(total_new_id);
#sc_mat = sc_mat[new_id][:, new_id];

np.save('../raw_data/subjet_12_intra_area_strengthen_false_sc_mat.npy',sc_mat);


#plt.imshow(sc_mat);

#plt.show();
