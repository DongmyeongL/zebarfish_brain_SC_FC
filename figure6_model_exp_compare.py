import matplotlib.pyplot as plt
import pickle
import numpy as np
import generate_sc_optimization as gso
from scipy.stats import linregress

load_file_name=[];
load_file_name.append('C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/subject_12_simulation_calcium_data_synaptic_weight_low.npz.npy');
load_file_name.append('C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/subject_12_simulation_calcium_data_synaptic_weight_high.npz.npy');
load_file_name.append('C:/Users/044ap/source/repos/MONET_SNN_CUDA_PYTHON_BOOST/x64/Release/subject_12_simulation_calcium_data_base.npz.npy');

                
save_file_name=[]
save_file_name.append('../raw_data/subject12_synaptic_weight_low_fg.npz');
save_file_name.append('../raw_data/subject12_synaptic_weight_high_fg.npz');   
save_file_name.append('../raw_data/subject12_base_fg.npz');       

tn=2;
tsave_file_name=save_file_name[tn];
emp_data=np.load('../raw_data/subject12_cellular_data_mean_spot_data.npy');

sim_data=np.load(load_file_name[tn]); #np.load('../raw_data/subject_12_simulation_calcium_data_base.npy');

mag=1.0;
plt.rcParams['font.size'] = 6*mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7*mag, 7*mag), dpi=300)

dd=0.75/4;
margin=0.05;
axes=[];
axes.append(fig.add_axes([margin,margin,dd,dd]));
axes.append(fig.add_axes([margin+dd+0.06,margin,dd,dd]));
axes.append(fig.add_axes([margin+(dd+0.06)*2,margin,dd,dd]));
axes.append(fig.add_axes([margin+(dd+0.06)*3,margin,dd,dd]));

hh=2*dd+0.06;
axes.append(fig.add_axes([margin,margin+dd+0.06,hh,dd]));
axes.append(fig.add_axes([margin+(dd+0.06)*2,margin+dd+0.06,hh,dd]));


axes.append(fig.add_axes([margin,margin+2*dd+0.12,hh,dd*2]));
axes.append(fig.add_axes([margin+(dd+0.06)*2,margin+2*dd+0.12,hh,dd*2]));



for i in range(len(emp_data)):
    axes[6].plot(emp_data[i]-i*2.0, color='black', alpha=0.7, linewidth=0.3); 
axes[6].axis('tight')
axes[6].axis('off')

for i in range(len(sim_data)):
    axes[7].plot(sim_data[i]*0.5-i*2.0, color='black', alpha=0.7, linewidth=0.3); 
axes[7].axis('tight')
axes[7].axis('off')


fc_vectors,emp_fc_mat,emp_whole_fc_mat=gso.fun_fc_vector((emp_data), 15,5);
fc_vectors,sim_fc_mat,sim_whole_fc_mat=gso.fun_fc_vector((sim_data), 10,5);

im1=axes[0].imshow(np.abs(emp_whole_fc_mat), cmap='jet', vmin=0, vmax=1);
axes[0].set_xticks([0,10,20,30,40,50,len(emp_whole_fc_mat)-1])
axes[0].set_yticks([0,10,20,30,40,50,len(emp_whole_fc_mat)-1])
plt.colorbar(im1, ax=axes[0])
axes[0].axis('tight')

im2=axes[1].imshow(np.abs(sim_whole_fc_mat), cmap='jet', vmin=0, vmax=1);
axes[1].set_xticks([0,10,20,30,40,len(emp_whole_fc_mat)-1])
axes[1].set_yticks([0,10,20,30,40,len(emp_whole_fc_mat)-1])
plt.colorbar(im2, ax=axes[1]);
axes[1].axis('tight')

flat_matrix1 = np.abs(sim_whole_fc_mat.flatten())
flat_matrix2 = np.abs(emp_whole_fc_mat.flatten())

ii=flat_matrix1>0
x_data=flat_matrix1[ii];
y_data=flat_matrix2[ii];
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
trendline_y = slope * x_data + intercept

#plt.figure();
#plt.scatter(x_data, y_data, s=1.0, alpha=0.7);plt.xlim(0, 1);plt.ylim(0, 1);
#plt.plot(x_data, trendline_y, color='red', linestyle='--', label=f'R={r_value:.2f}')


#plt.xlabel('Simulation FC');
#plt.ylabel('Empirical FC');
#plt.legend();
#plt.axis('tight')

eg1,eg2=gso.fun_fg(emp_fc_mat);
sg1,g2=gso.fun_fg(sim_fc_mat);  
    

im3=axes[4].imshow(eg1.T, cmap='jet',vmin=-0.2,vmax=0.3);
plt.colorbar(im3, ax=axes[4]);
axes[4].axis('tight');
im4=axes[5].imshow(sg1.T, cmap='jet',vmin=-0.2,vmax=0.3);
plt.colorbar(im4,ax=axes[5]);
axes[5].axis('tight')



x_mean = np.mean(eg1, axis=0)
y_mean = np.mean(sg1, axis=0)

x_std = np.std(eg1, axis=0)
y_std = np.std(sg1, axis=0)



axes[2].plot(x_mean[0:20],'.-',label='Empirical Mean', color='blue');
axes[2].plot(y_mean[0:20],'.-',label='Simulation Mean', color='red');
axes[2].set_xticks([0,5,10,15,20])
axes[2].axis('tight')

axes[3].plot(x_std[0:20],'.-',label='Empirical Mean', color='blue');
axes[3].plot(y_std[0:20],'.-',label='Simulation Mean', color='red');
axes[2].set_xticks([0,5,10,15,20])
#plt.title('STD of 1st Functional Gradients');
#plt.legend();
plt.axis('tight')

for i in range(8):
    axes[i].tick_params(axis='x', pad=1, length=1)
    axes[i].tick_params(axis='y', pad=1, length=1)
    
fig.savefig('./fig6.eps',format='eps',dpi=300,transparent=True);    

plt.show();
#np.savez(tsave_file_name,x_mean=x_mean, y_mean=y_mean, x_std=x_std, y_std=y_std, emp_fc_mat=emp_fc_mat, sim_fc_mat=sim_fc_mat);
  
plt.show();