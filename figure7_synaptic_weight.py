import matplotlib.pyplot as plt
import numpy as np
import generate_sc_optimization as gso
import matplotlib as mpl
mpl.rcParams['path.simplify'] = False

# ---------- 설정 ----------
mag = 1.0
plt.rcParams['font.size'] = 6 * mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7 * mag, 5 * mag), dpi=300)

dd = 0.8 / 2
margin = 0.05
ddd = dd / 2.2

# ---------- 축 생성 ----------
axes = [
    fig.add_axes([margin, margin, dd, dd]),
    fig.add_axes([margin + dd + 0.1, margin, dd, dd]),
    fig.add_axes([margin + dd + 0.1, margin + dd + 0.05, ddd, ddd]),
    fig.add_axes([margin + dd + 0.12 + dd / 2, margin + dd + 0.05, ddd, ddd]),
    fig.add_axes([margin + dd + 0.1, margin + dd + 0.1 + ddd, ddd, ddd]),
    fig.add_axes([margin + dd + 0.12 + dd / 2, margin + dd + 0.1 + ddd, ddd, ddd]),
]

# ---------- 유틸리티 함수 ----------
def process_data(file_template, n_files=10):
    mean_list, std_list = [], []
    for i in range(n_files):
        data = np.load(file_template.format(i))
        fc_vectors, sim_fc, sim_whole_fc = gso.fun_fc_vector(data, 10, 5)
        sg1, _ = gso.fun_fg(sim_fc)
        mean_list.append(np.mean(sg1, axis=0))
        std_list.append(np.std(sg1, axis=0))
    return np.array(mean_list), np.array(std_list)

# ---------- 데이터 불러오기 ----------
base_path = "C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/"
low_path = base_path + "subject_12_base_sc_for_simualtion_low_variance_ca_data_{}.npy"
high_path = base_path + "subject_12_base_sc_for_simualtion_high_variance_ca_data_{}.npy"
ref_path = base_path + "subject_12_base_sc_for_simualtion_ca_data_{}.npy"

m1, s1 = process_data(low_path)
m2, s2 = process_data(high_path)
m3, s3 = process_data(ref_path)

def plot_aggregate_shading(ax, data, color, label):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)/np.sqrt(9)
    
   
    x_vals = np.arange(data.shape[1])
    
    
    
    ax.plot(x_vals, mean,'o-', markersize=3,color=color, linewidth=1, label=label)
    ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.3)
   
for i in range(10):
    for j in range(20):
        s1[i][j]=s1[i][j]/np.abs(m1[i][j]);        
for i in range(10):
    for j in range(20):
        s2[i][j]=s2[i][j]/np.abs(m2[i][j])
for i in range(10):
    for j in range(20):
        s3[i][j]=s3[i][j]/np.abs(m3[i][j])

                  
    
plot_aggregate_shading(axes[0],m1[:,0:20],'#B0C4DE','Low Synaptic Weight');
plot_aggregate_shading(axes[0],m2[:,0:20],'red','High Synaptic Weight');
plot_aggregate_shading(axes[0],m3[:,0:20],'black','Reference');

# ---------- plot: 평균 ----------
#axes[0].plot(m1[:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight')
#axes[0].plot(m2[:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight')
#axes[0].plot(m3[:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight')
axes[0].set_xticks(np.arange(20))
axes[0].axis('tight')
axes[0].legend()


    
    
plot_aggregate_shading(axes[1],s1[:,0:20],'blue','Low Synaptic Weight');
plot_aggregate_shading(axes[1],s2[:,0:20],'red','High Synaptic Weight');
plot_aggregate_shading(axes[1],s3[:,0:20],'black','Reference');


# ---------- plot: 표준편차 ----------
#axes[1].plot(s1[:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight')
#axes[1].plot(s2[:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight')
#axes[1].plot(s3[:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight')
axes[1].set_xticks(np.arange(20))
axes[1].axis('tight')
axes[1].legend()

# ---------- FC 및 FG 매트릭스 시각화 ----------
def plot_matrix_and_fg(ax_fc, ax_fg, file_path, vmin_fc=0, vmax_fc=1, vmin_fg=0, vmax_fg=1):
    sim_data = np.load(file_path)
    _, sim_fc, sim_whole_fc = gso.fun_fc_vector(sim_data, 10, 5)
    sg1, _ = gso.fun_fg(sim_fc)

    im_fc = ax_fc.imshow(np.abs(sim_whole_fc), cmap='jet', vmin=vmin_fc, vmax=vmax_fc)
    plt.colorbar(im_fc, ax=ax_fc, fraction=0.046)
    ax_fc.set_xticks([0, 10, 20, 30, 40, 50, len(sim_whole_fc) - 1])
    ax_fc.set_yticks([0, 10, 20, 30, 40, 50, len(sim_whole_fc) - 1])
    #ax_fc.axis('tight')

    im_fg = ax_fg.imshow(sg1.T, cmap='jet', vmin=vmin_fg, vmax=vmax_fg)
    plt.colorbar(im_fg, ax=ax_fg, fraction=0.2)
    #ax_fg.axis('tight')

# 특정 인덱스 예시 (i=3)
i = 2
plot_matrix_and_fg(axes[4], axes[2], low_path.format(i))
plot_matrix_and_fg(axes[5], axes[3], high_path.format(i))

# ---------- 축 설정 ----------
for ax in axes:
    ax.tick_params(axis='x', pad=1, length=1)
    ax.tick_params(axis='y', pad=1, length=1)

# ---------- 저장 및 출력 ----------
fig.savefig('./fig7.svg', format='svg', dpi=300, transparent=True)
plt.show()


