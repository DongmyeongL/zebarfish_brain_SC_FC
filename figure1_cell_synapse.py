import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from random import uniform
from colorsys import hsv_to_rgb
from scipy.stats import linregress

def generate_n_colors(nColors, saturation=65, value=90, randomness=0):
    """
    Generates a list of distinct RGB colors.

    Args:
        nColors (int): The number of colors to generate.
        saturation (int): Base saturation percentage (0-100).
        value (int): Base value percentage (0-100).
        randomness (float): Amount of randomness to add to saturation.

    Returns:
        list: A list of RGB color tuples.
    """
    h = np.linspace(0, 320, nColors)
    s = np.array([saturation + uniform(-randomness, randomness)] * nColors)
    v = np.array([value] * nColors)
    palette = []
    for i in range(nColors):
        palette.append(hsv_to_rgb(h[i] / 360, s[i] / 100, v[i] / 100))
    return palette

def resize_matrix_average_pooling(matrix, target_size=70):
    
    """
    평균 풀링을 사용하여 행렬의 크기를 target_size x target_size로 조정합니다.
    """
    original_size = matrix.shape[0]
    if original_size < target_size:
        raise ValueError("원본 행렬 크기는 목표 크기보다 작을 수 없습니다.")

    # 풀링할 블록의 크기 계산
    block_size = original_size // target_size
    
    # 목표 크기에 맞게 원본 행렬을 자름 (나머지 부분은 버림)
    # 이는 깔끔한 분할을 위해 필요합니다.
    trimmed_size = block_size * target_size
    trimmed_matrix = matrix[:trimmed_size, :trimmed_size]

    # Reshaping을 통해 평균 풀링 수행
    # (원본 크기 / 목표 크기, 목표 크기, 원본 크기 / 목표 크기, 목표 크기)로 재구성 후 평균
    reshaped_matrix = trimmed_matrix.reshape(
        target_size, block_size, target_size, block_size
    )
    pooled_matrix = reshaped_matrix.mean(axis=(1, 3))
    
    return pooled_matrix

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


mag=1.0;
plt.rcParams['font.size'] = 6*mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7*mag, 7*mag), dpi=300)

dd=0.75/4;
margin=0.05;
axes=[];
axes.append(fig.add_axes([margin,margin,dd,dd]));
axes.append(fig.add_axes([margin+(margin+dd)*1,margin,dd,dd]));
axes.append(fig.add_axes([margin+(margin+dd)*2,margin,dd,dd]));
axes.append(fig.add_axes([margin+(margin+dd)*3,margin,dd,dd]));

h1=margin+dd+margin;
axes.append(fig.add_axes([0.05,h1,0.8/3,0.8/3]));
axes.append(fig.add_axes([0.1+0.8/3,h1,0.8/3,0.8/3]));
axes.append(fig.add_axes([0.15+0.8/3*2,h1,0.8/3,0.8/3]));


h2=h1+0.8/3+margin
w2=1-h2-0.05;
axes.append(fig.add_axes([0.05,h2,0.8/3,w2]));
axes.append(fig.add_axes([0.8/3+0.1,h2,0.8/3,w2]));



file_name=f'../raw_data/subject_12_data_synapse_sc_aux_data.pkl';
with open(file_name, 'rb') as f:
        load_data = pickle.load(f)

t_CellXYZ_full=load_data['t_CellXYZ_full'];
indices_ii=load_data['indices_ii'];
syn_xyz=load_data['syn_xyz']
syn_idx=load_data['syn_idx'];
t_spot_data=load_data['t_spot_data'];
re_sc_mat=load_data['re_sc_mat'];
synapse_region_mean=load_data['synapse_region_mean'];

n_region=len(indices_ii);
palette = generate_n_colors(n_region)
#random.shuffle(palette) # Shuffle colors for better visual distinction
for j, group in enumerate(indices_ii):
        axes[7].scatter(t_CellXYZ_full[group, 1], t_CellXYZ_full[group, 0], color=palette[j],
                                marker='.', edgecolor='none', alpha=0.2, s=3, rasterized=True)
axes[7].set_ylim([950,100])
axes[7].set_xlim([0,600])         
#axes[6].set_facecolor('black');
axes[7].axis('off')

sy_max=np.max(syn_idx)+1;
axes[8].scatter(t_CellXYZ_full[:, 1], t_CellXYZ_full[:, 0], c=palette[0],marker='s', edgecolor='none', alpha=0.1, s=0.5, rasterized=True)

#t_palette = generate_n_colors(sy_max);
#syn_palette=[];
#for i in range(len(syn_idx)):
#    syn_palette.append(t_palette[syn_idx[i]]);
    
axes[8].scatter(syn_xyz[:, 1], syn_xyz[:, 0], c=palette[24], marker='^', edgecolor='none', alpha=0.2, s=0.5, rasterized=True)
axes[8].set_ylim([950,100])
axes[8].set_xlim([0,600])   
#axes[7].set_facecolor('black');
axes[8].axis('off')

raw_ca_data=np.load('../raw_data/subject_12_simulation_base_raw_ca_data.npy');



'''
sc_mat=np.load('../raw_data/subject_12_data_synapse_sc_mat.py.npy', allow_pickle=True)
fc_mat= np.corrcoef(t_spot_data)
# Plotting matrices on axes[4] and axes[5]
ax4 = axes[4]
ax4.imshow(sc_mat, cmap='binary')
#ax4.set_xticks([]) # Hide x-axis ticks
#ax4.set_yticks([]) # Hide y-axis ticks
ax4.set_title('SC Mat') # Optional: Add a title for clarity

ax5 = axes[5]
im = ax5.imshow(np.abs(fc_mat), cmap='jet',vmin=0.0,vmax=0.4)
plt.colorbar(im, ax=ax5) # More controlled colorbar placement
#im.set_clim([0.0, 1.0])
#ax5.set_xticks([]) # Hide x-axis ticks
#ax5.set_yticks([]) # Hide y-axis ticks
ax5.set_title('FC Mat') # Optional: Add a title for clarity
print(sc_mat.shape)
fc_mat=0;

raw_ca_data_fc=(np.abs(np.corrcoef(raw_ca_data)));
ax6 = axes[6]
im = ax6.imshow(raw_ca_data_fc, cmap='jet',vmin=0.0,vmax=1.0)
plt.colorbar(im, ax=ax6)
'''

re_fc_mat=np.corrcoef(synapse_region_mean);


im1=axes[0].imshow(np.log10(re_sc_mat+1), cmap='jet');
plt.colorbar(im1,ax=axes[0],shrink=0.8) # More controlled colorbar placement
#fig.colorbar(im1,ax=axes[0]);
axes[0].set_xticks([0,10,20,30,40,50]) # Hide x-axis ticks
axes[0].set_yticks([0,10,20,30,40,50]) # Hide y-axis ticks

for i in range(len(re_fc_mat)):
    re_fc_mat[i,i]=0;
im2=axes[1].imshow(np.abs(re_fc_mat),cmap='jet');
plt.colorbar(im2, ax=axes[1],shrink=0.8) # More controlled colorbar placement

#fig.colorbar(im2,ax=axes[1]);
axes[1].set_xticks([0,10,20,30,40,50]) # Hide x-axis ticks
axes[1].set_yticks([0,10,20,30,40,50] ) # Hide y-axis ticks



flat_matrix1 = re_sc_mat.flatten()
flat_matrix2 = re_fc_mat.flatten()

ii=flat_matrix1>0
x_data=np.log10(flat_matrix1[ii]+1);
y_data=flat_matrix2[ii];

from scipy.stats import linregress
axes[2].scatter(np.log10(flat_matrix1[ii]+1),flat_matrix2[ii],s=0.05,alpha=0.7);
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
trendline_y = slope * x_data + intercept
axes[2].plot(x_data, trendline_y, color='red', linestyle='--', label=f'R={r_value:.2f})')
axes[2].set_xticks([1,2,3,4,5])
axes[2].axis('tight')

#axes[2].legend() # 라벨 표시
#axes[2].set_in_layout() # 

re_sc_fc_corr_data=[];
re_sc_flat_data=[];
for file_idx in [12,13,14,15,16,17,18]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    re_sc_mat=load_data['re_sc_mat'];
    synapse_region_mean=load_data['synapse_region_mean'];
    re_fc_mat=np.corrcoef(synapse_region_mean);
    flat_matrix1 = re_sc_mat.flatten()
    flat_matrix2 = re_fc_mat.flatten()

    re_sc_flat_data.append(flat_matrix1);

    ii=flat_matrix1>0
    x_data=np.log10(flat_matrix1[ii]+1);
    y_data=flat_matrix2[ii];
    
    corr=np.corrcoef(x_data,y_data);
    re_sc_fc_corr_data.append(corr[0,1]);
    print(file_idx,corr[0,1]);


check_region=np.zeros([7,72]);

jj=0;
for file_idx in [12,13,14,15,16,17,18]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    
    for ii in synapse_id:
        check_region[jj,ii]=1;
    jj+=1;    


sum_check_region=np.sum(check_region,axis=0);

check_region_num=[];
for i in range(72):
        if(sum_check_region[i]==7):
            check_region_num.append(i);



re_sc_flat_mat=[];
for file_idx in [12,13,14,15,16,17,18]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    new_synapse_id=[];
    
    l=0;
    for k in check_region_num:
        try:
            kk=synapse_id.index(k);
            new_synapse_id.append(kk);
        except:
             a=1;
    
    re_sc_mat=load_data['re_sc_mat'];
    
    new_re_sc_mat=re_sc_mat[np.ix_(new_synapse_id, new_synapse_id)];
    flat_matrix1 =new_re_sc_mat.flatten();
    re_sc_flat_mat.append(flat_matrix1);
    aaa=1;        
            
mag=1/4
fig = plt.figure(figsize=(7*mag, 7*mag), dpi=300)
dd=0.75;
hh1=0.85
margin=0.05;
n_axes=[];

n_axes.append(fig.add_axes([margin+0.05,margin+0.05,dd,hh1]));          
im=n_axes[0].imshow(np.corrcoef(re_sc_flat_mat),cmap='jet');
n_axes[0].set_xticks([0,1,2,3,4,5,6])
n_axes[0].set_yticks([0,1,2,3,4,5,6])
plt.colorbar(im,ax=n_axes[0],shrink=0.7);


for ax in n_axes:
    ax.tick_params(axis='x', pad=1, length=1)
    ax.tick_params(axis='y', pad=1, length=1)
    
fig.savefig('./fig1-1.svg',format='svg',dpi=300,transparent=True);

    
plt.show();    

      
aaa=1;
    
re_sc_fc_corr_data=np.array(re_sc_fc_corr_data);

mean_val = np.mean(re_sc_fc_corr_data)
std_val = np.std(re_sc_fc_corr_data, ddof=1)


from scipy.stats import sem

sem_val = sem(re_sc_fc_corr_data);

x_pos=1;
jittered_x = np.random.normal(loc=x_pos, scale=0.05, size=len(re_sc_fc_corr_data))
x_pos = np.mean(jittered_x)

axes[3].scatter(jittered_x, re_sc_fc_corr_data, color='blue', alpha=0.7, label='Data points')
axes[3].errorbar(x_pos, mean_val, yerr=sem_val, fmt='.', color='black', capsize=3, label='Mean ± SEM')
axes[3].set_xlim(np.min(jittered_x)-0.1, np.max(jittered_x)+0.1);
axes[3].set_ylim(0.2, 0.8);


for ax in axes:
    ax.tick_params(axis='x', pad=1, length=1)
    ax.tick_params(axis='y', pad=1, length=1)
    


fig.savefig('./fig1.svg',format='svg',dpi=300,transparent=True);




#plt.show();