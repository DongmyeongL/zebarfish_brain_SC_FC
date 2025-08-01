import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from random import uniform
from colorsys import hsv_to_rgb
from scipy.stats import linregress
import generate_sc_optimization as gso


x_data=[];
y_data=[];

for file_idx in [12,13,14,15,16,17,18]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    
    re_sc_mat=load_data['re_sc_mat'];
    synapse_region_mean=load_data['synapse_region_mean'];
    synapse_region_mean=np.array(synapse_region_mean)
    fc_vectors,emp_fc_mat,emp_whole_fc_mat=gso.fun_fc_vector((synapse_region_mean), 15,5);
    
    tx=np.mean(fc_vectors,axis=1);
    for j in range(len(emp_fc_mat)):
        flat_mat1 = emp_fc_mat[j].flatten();
        flat_mat2 = (re_sc_mat.flatten());

    
        ii=flat_mat2>0
        xt=np.log10(flat_mat2[ii]+1);
        yt=flat_mat1[ii]
        cc=np.corrcoef(xt,yt);
        x_data.append(tx[j]);
        y_data.append(cc[0,1]);
        
    
x_data=np.array(x_data);
y_data=np.array(y_data);

slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
trendline_y = slope * x_data + intercept

# ---------- 설정 ----------
mag = 1.0
plt.rcParams['font.size'] = 6 * mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7 * mag, 5 * mag), dpi=300)

dd=0.75/4;
hh1=0.85/2-0.05;
margin=0.05;
axes=[];

axes.append(fig.add_axes([margin,margin+0.05,2*dd,hh1]));

axes.append(fig.add_axes([margin,margin+0.15+hh1,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*1,margin+0.15+hh1,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*2,margin+0.15+hh1,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*3,margin+0.15+hh1,dd,hh1]));
                          


axes[0].scatter(x_data,y_data,s=1);    
axes[0].plot(x_data, trendline_y, color='red', linestyle='--', label=f'R={r_value:.2f}')
axes[0].legend();   

for file_idx in [12]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    
    re_sc_mat=load_data['re_sc_mat'];
    
    
    synapse_region_mean=load_data['synapse_region_mean'];
    synapse_region_mean=np.array(synapse_region_mean)
    
    fc_vectors,emp_fc_mat,emp_whole_fc_mat=gso.fun_fc_vector((synapse_region_mean), 15,5);

    tx=np.mean(fc_vectors,axis=1);
   
    
    im=axes[1].imshow(np.log10(re_sc_mat+1),cmap='jet');
    plt.colorbar(im,ax=axes[1],shrink=0.7);
    
    im=axes[2].imshow(np.abs(emp_fc_mat[18]),cmap='jet');
    plt.colorbar(im,ax=axes[2],shrink=0.7);
    
    im=axes[3].imshow(np.abs(emp_fc_mat[57]),cmap='jet');
    plt.colorbar(im,ax=axes[3],shrink=0.7);
    
    im=axes[4].imshow(np.abs(emp_fc_mat[93]),cmap='jet');
    plt.colorbar(im,ax=axes[4],shrink=0.7);
    
    

for ax in axes:
    ax.tick_params(axis='x', pad=1, length=1)
    ax.tick_params(axis='y', pad=1, length=1)
    
        
fig.savefig('./fig9.svg',format='svg',dpi=300,transparent=True);


 
plt.show();

