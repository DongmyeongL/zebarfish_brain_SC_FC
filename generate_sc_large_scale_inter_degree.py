import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def analyze_laplacian(sc_matrix, plot=True):
    """
    SC 행렬에서 Laplacian 행렬을 생성하고, eigenvalue/eigenvector 분석을 수행합니다.

    Parameters:
    -----------
    sc_matrix : ndarray
        구조 연결성(Structural Connectivity) 행렬 (대칭, 양의 실수)
    plot : bool
        고유값 및 주요 고유벡터 시각화 여부

    Returns:
    --------
    L : ndarray
        Laplacian matrix
    eigvals : ndarray
        고유값 (오름차순 정렬)
    eigvecs : ndarray
        고유벡터 (열 벡터 기준)
    """

    # Degree matrix
    degrees = np.sum(sc_matrix, axis=1)
    D = np.diag(degrees)

    # Laplacian: L = D - A
    L = D - sc_matrix

    # 고유값 및 고유벡터 계산 (symmetric → eigh 사용)
    eigvals, eigvecs = eigh(L)

    if plot:
        plt.figure(figsize=(12, 4))

        # 고유값 플롯
        plt.subplot(1, 2, 1)
        plt.plot(eigvals, 'o-')
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.title("Laplacian Spectrum")

        # 두 번째 고유벡터 (Fiedler vector)
        plt.subplot(1, 2, 2)
        
       # plt.plot(eigvecs[:, 100], 'r.-')
        plt.imshow(np.abs(eigvecs));
        plt.xlabel("Node Index")
        plt.ylabel("Value")
        plt.title("2nd Eigenvector (Fiedler vector)")

        plt.tight_layout()


    return L, eigvals, eigvecs

def detect_core_nodes(SC, core_ratio=0.25):
    degrees = SC.sum(axis=0)
    num_core = max(1, int(len(degrees) * core_ratio))
    return list(np.argsort(degrees)[-num_core:])

def modify_interarea_core_periphery(SC, area1, area2, core_ratio=0.25, strengthen=True, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    core1 = {area1[i] for i in detect_core_nodes(SC[np.ix_(area1, area1)], core_ratio)}
    core2 = {area2[i] for i in detect_core_nodes(SC[np.ix_(area2, area2)], core_ratio)}
    periph1 = set(area1) - core1
    periph2 = set(area2) - core2

    inter_edges = [(i, j) for i in area1 for j in area2 if SC[i, j] > 0]

    cc_edges, pp_edges, cp_edges = [], [], []
    for i, j in inter_edges:
        if (i in core1 and j in core2) or (i in core2 and j in core1):
            cc_edges.append((i, j))
        elif (i in periph1 and j in periph2) or (i in periph2 and j in periph1):
            pp_edges.append((i, j))
        else:
            cp_edges.append((i, j))

    if strengthen and pp_edges:
        n = int(len(pp_edges) * 0.6)
        random.shuffle(pp_edges)
        added = list(zip(
            random.choices(tuple(core1), k=n),
            random.choices(tuple(core2), k=n)
        ))
        pp_edges = pp_edges[n:]
        cc_edges.extend(added)

    elif not strengthen and cc_edges and len(periph1) > 10 and len(periph2) > 10:
        n = int(len(cc_edges) * 0.3)
        random.shuffle(cc_edges)
        added = list(zip(
            random.choices(tuple(periph1), k=n),
            random.choices(tuple(periph2), k=n)
        ))
        cc_edges = cc_edges[n:]
        pp_edges.extend(added)

    new_SC = np.zeros_like(SC)
    for i, j in cc_edges + cp_edges + pp_edges:
        new_SC[i, j] = new_SC[j, i] = 1
    new_SC[np.ix_(area1, area1)] = SC[np.ix_(area1, area1)]
    new_SC[np.ix_(area2, area2)] = SC[np.ix_(area2, area2)]
    return new_SC

# ---------------- 데이터 로딩 -----------------
data = np.load('../raw_data/subject12_cellular_data.npz', allow_pickle=True)
sc_mat = data['sc_mat']
with open("../raw_data/subject12_cellular_data_sc_mat_indices.pkl", "rb") as f:
    indices_ii = pickle.load(f)

# ---------------- 영역 내부 정렬 최적화 -----------------
total_indices = np.concatenate(indices_ii)
sorted_indices = []

for inds in indices_ii:
    sub = sc_mat[np.ix_(inds, inds)]
    local_order = np.argsort(np.sum(sub, axis=0))
    sorted_indices.extend(np.array(inds)[local_order])

sorted_indices = np.array(sorted_indices)

#sc_mat = sc_mat[sorted_indices][:, sorted_indices]

# ---------------- 영역 간 rewiring -----------------

from joblib import Parallel, delayed

def process_interarea(i, j, indices_ii, sc_mat, strengthen):
    
    #print(i,j)
    pick_i = np.array(indices_ii[i])
    pick_j = np.array(indices_ii[j])
    total_pick = np.concatenate([pick_i, pick_j])
    offset_i = np.arange(len(pick_i))
    offset_j = np.arange(len(pick_j)) + len(pick_i)

    sub_sc = sc_mat[np.ix_(total_pick, total_pick)]
    modified = modify_interarea_core_periphery(sub_sc, offset_i, offset_j, strengthen=strengthen)
    return total_pick, modified

'''
ii=20;
jj=22;
total_pick, new_sc_mat=process_interarea(ii, jj, indices_ii, sc_mat, strengthen=False);

plt.figure();
plt.subplot(1,2,1);plt.imshow(sc_mat[np.ix_(total_pick, total_pick)]);
plt.subplot(1,2,2);plt.imshow(new_sc_mat);
plt.show();

aaa;
'''

#sub 영역 분석



# 병렬 실행 (n_jobs=-1: 사용 가능한 모든 CPU 코어 사용)
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_interarea)(i, j, indices_ii, sc_mat, False)
    for i in range(len(indices_ii))
    for j in range(i+1, len(indices_ii))
)

# 결과를 sc_mat에 반영
for pick_indices, modified_sub_sc in results:
    sc_mat[np.ix_(pick_indices, pick_indices)] = modified_sub_sc
    
'''    
ll=len(indices_ii);
for i in range(ll):
    for j in range(i+1, ll):
        pick_i = np.array(indices_ii[i])
        pick_j = np.array(indices_ii[j])
        total_pick = np.concatenate([pick_i, pick_j])

        offset_i = np.arange(len(pick_i))
        offset_j = np.arange(len(pick_j)) + len(pick_i)

        sub_sc = sc_mat[np.ix_(total_pick, total_pick)]
        modified = modify_interarea_core_periphery(sub_sc, offset_i, offset_j, strengthen=True)
        sc_mat[np.ix_(total_pick, total_pick)] = modified
'''

# ---------------- 저장 및 시각화 -----------------
np.save('../raw_data/subjet_12_inter_area_strengthen_false_sc_mat.npy', sc_mat)

aaa;


plt.figure(figsize=(8, 8))
plt.imshow(sc_mat, cmap='gray_r')
plt.title('Modified SC matrix')
plt.show()


