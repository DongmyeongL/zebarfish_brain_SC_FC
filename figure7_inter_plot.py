import matplotlib.pyplot as plt
import numpy as np
import generate_sc_optimization as gso

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
low_path = base_path + "subject_12_base_sc_for_simualtion_inter_strengthen_false_ca_data_{}.npy"
high_path = base_path + "subject_12_base_sc_for_simualtion_inter_strengthen_true_ca_data_{}.npy"
ref_path = base_path + "subject_12_base_sc_for_simualtion_ca_data_{}.npy"

m1, s1 = process_data(low_path)
m2, s2 = process_data(high_path)
m3, s3 = process_data(ref_path)

def plot_aggregate_shading(ax, data, color, label):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)/np.sqrt(9)
    x_vals = np.arange(data.shape[1])
    
    ax.plot(x_vals, mean, color=color, linewidth=2.0, label=label)
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



    
plot_aggregate_shading(axes[0],m1[:,0:20],'blue','P-P');
plot_aggregate_shading(axes[0],m2[:,0:20],'red','C-C');
plot_aggregate_shading(axes[0],m3[:,0:20],'black','Reference');

# ---------- plot: 평균 ----------
#axes[0].plot(m1[:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight')
#axes[0].plot(m2[:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight')
#axes[0].plot(m3[:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight')
axes[0].set_xticks(np.arange(20))
axes[0].axis('tight')
axes[0].legend()


    
    
plot_aggregate_shading(axes[1],s1[:,0:20],'blue','P-P');
plot_aggregate_shading(axes[1],s2[:,0:20],'red','C-C');
plot_aggregate_shading(axes[1],s3[:,0:20],'black','Reference');


# ---------- plot: 표준편차 ----------
#axes[1].plot(s1[:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight')
#axes[1].plot(s2[:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight')
#axes[1].plot(s3[:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight')
axes[1].set_xticks(np.arange(20))
axes[1].axis('tight')
axes[1].legend()

# ---------- FC 및 FG 매트릭스 시각화 ----------
def plot_matrix_and_fg(ax_fc, ax_fg, file_path, vmin_fc=0, vmax_fc=1, vmin_fg=-0.2, vmax_fg=0.3):
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
fig.savefig('./fig8.eps', format='eps', dpi=300, transparent=True)
plt.show()






aaaaaa;
import matplotlib.pyplot as plt
import pickle
import numpy as np
import generate_sc_optimization as gso
from scipy.stats import linregress

          
          
mag=1.0;
plt.rcParams['font.size'] = 6*mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7*mag, 5*mag), dpi=300)

dd=0.8/2;
margin=0.05;

axes=[];
axes.append(fig.add_axes([margin,margin,dd,dd]));
axes.append(fig.add_axes([margin+dd+0.1,margin,dd,dd]));

ddd=dd/2.2
axes.append(fig.add_axes([margin+dd+0.1,margin+dd+0.05,ddd,ddd]));
axes.append(fig.add_axes([margin+dd+0.12+dd/2,margin+dd+0.05,ddd,ddd]));

axes.append(fig.add_axes([margin+dd+0.1,margin+dd+0.1+ddd,ddd,ddd]));
axes.append(fig.add_axes([margin+dd+0.12+dd/2,margin+dd+0.1+ddd,ddd,ddd]));

load_file_name=[];
load_file_name.append('C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/subject_12_simulation_calcium_data_inter_false.npz.npy');
load_file_name.append('C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/subject_12_simulation_calcium_data_inter_true.npz.npy');

qq=[4,5];
for ii in range(2):

    sim_data=np.load(load_file_name[ii]);
    
    fc_vectors,sim_fc_mat,sim_whole_fc_mat=gso.fun_fc_vector((sim_data), 10,5);
    
    #plt.figure();
    im=axes[qq[ii]].imshow(np.abs(sim_whole_fc_mat), cmap='jet', vmin=0, vmax=1);
    plt.colorbar(im,ax=axes[qq[ii]]);
    axes[qq[ii]].set_xticks([0,10,20,30,40,50,len(sim_whole_fc_mat)-1])
    axes[qq[ii]].set_yticks([0,10,20,30,40,50,len(sim_whole_fc_mat)-1])
    axes[qq[ii]].axis('tight');
    sg1,g2=gso.fun_fg(sim_fc_mat);
    im3=axes[qq[ii]-2].imshow(sg1.T, cmap='jet',vmin=-0.2,vmax=0.3);
    plt.colorbar(im3,ax=axes[qq[ii]-2],fraction=0.2);
    axes[qq[ii]].axis('tight');


       
save_file_name=[]
save_file_name.append('../raw_data/subject12_inter_false_fg.npz');
save_file_name.append('../raw_data/subject12_inter_true_fg.npz');   
save_file_name.append('../raw_data/subject12_base_fg.npz');



load_file_name=save_file_name[0];

data1 = np.load(save_file_name[0]);
data2 = np.load(save_file_name[1]);
data3 = np.load(save_file_name[2]);

m1 = data1['y_mean']
m2 = data2['y_mean']        
m3 = data3['y_mean']


s1 = data1['y_std']
s2 = data2['y_std']        
s3 = data3['y_std']

axes[0].plot(m1[0:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight');
axes[0].plot(m2[0:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight');
axes[0].plot(m3[0:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight');    

axes[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
axes[0].axis('tight')
axes[0].legend();


axes[1].plot(s1[0:20], color='blue', alpha=0.7, linewidth=2.5, label='Low Synaptic Weight');
axes[1].plot(s2[0:20], color='red', alpha=0.7, linewidth=2.5, label='High Synaptic Weight');
axes[1].plot(s3[0:20], color='black', alpha=0.7, linewidth=2.5, label='Base Synaptic Weight');    
axes[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
axes[1].axis('tight')
axes[1].legend();

for i in range(6):
    axes[i].tick_params(axis='x', pad=1, length=1)
    axes[i].tick_params(axis='y', pad=1, length=1)

    
    
    
fig.savefig('./fig8.eps',format='eps',dpi=300,transparent=True);    

plt.show();



aaaaa;
plt.figure();
plt.plot(m1[0:20], color='blue', alpha=0.7, linewidth=2.5, label='Peri.-Peri.');
plt.plot(m2[0:20], color='red', alpha=0.7, linewidth=2.5, label='Core-Core');
plt.plot(m3[0:20], color='black', alpha=0.7, linewidth=2.5, label='Base ');    
plt.title('Mean of 1st Functional Gradients');
plt.axis('tight')
plt.legend();




plt.figure();
plt.plot(s1[0:20], color='blue', alpha=0.7, linewidth=2.5, label='Peri.-Peri.');
plt.plot(s2[0:20], color='red', alpha=0.7, linewidth=2.5, label='Core-Core');
plt.plot(s3[0:20], color='black', alpha=0.7, linewidth=2.5, label='Base');    
plt.title('STD of 1st Functional Gradients');
plt.axis('tight')
plt.legend();

plt.show();
