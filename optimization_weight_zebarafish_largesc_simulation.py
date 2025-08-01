import numpy as np
import MONET_SNN_CUDA_PYTHON_BOOST as snn
import generate_sc_optimization as gso
import pickle
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real


def fun_sc_weight(xx):
    log_std, g_weight=xx;
    file_name1=f'C:/Users/044ap/OneDrive/2025_zebrafish_topology/heavy_tailed_topology_function/raw_data/subject12_cellular_data.npz';
    file_name2=f'C:/Users/044ap/OneDrive/2025_zebrafish_topology/heavy_tailed_topology_function/raw_data/subject12_cellular_data_sc_mat_indices.pkl';

    file_name=f'../raw_data/subject_12_base_sc_for_simualtion.pkl';
    with open(file_name, 'rb') as f:
        loaded_data = pickle.load(f)
    
    custom_connections=loaded_data['custom_connections'];
    connection_dist=loaded_data['connection_dist'];
    weight_dist=loaded_data['weight_dist']
    
    
    for i in range(len(weight_dist)):
        for j in range(len(weight_dist[i])):
            weight_dist[i][j]*=g_weight;
            log_mean=1.23*np.exp(-0.05*connection_dist[i][j])-5.5611;
        #log_std=1.0; #0.01,0.1,1.0;
            weights = np.random.lognormal(mean=log_mean, sigma=log_std)*g_weight;
            weight_dist[i][j]=weights;
            
    
    
    with open(file_name2, "rb") as f:
        indices_ii = pickle.load(f)
        
    '''    
    data = np.load(file_name1, allow_pickle=True)
    sel_neuron = np.array(data['sel_neuron']);
    sc_mat=np.array(data['sc_mat']);

    CellXYZ_data = np.array(data['t_CellXYZ_data'][sel_neuron]);
    CellXYZ_data=CellXYZ_data.astype(float);


    loaded_data=gso.generate_sc_weight(sc_mat,CellXYZ_data,log_std);    
    
    '''
    num_neurons=29317;
    inh_prob = 0.2
    #g_weight = 20.2
    final_time_steps = 100000    # Shorter for quick test; for 30k neurons, this is still substantial
    dt_val = 0.05
    ca_window = 500;
    noise_intensity = 1.5;
    '''
    custom_connections=loaded_data['custom_connections'];
    connection_dist=loaded_data['connection_dist'];
    weight_dist=loaded_data['weight_dist']


    for i in range(len(weight_dist)):
        for j in range(len(weight_dist[i])):
            weight_dist[i][j]*=g_weight;
    '''        
    simulator = snn.NeuronSimulator(
        num_neurons,
        noise_intensity,
        connection_dist,
        custom_connections,
        inh_prob,
        weight_dist,
        final_time_steps,
        dt_val,
        ca_window
    )

    simulator.run_simulation()

    calcium_data = simulator.get_calcium_data() # This now returns a Python list
    # You can directly convert the Python list to a NumPy array for reshaping/analysis
    calcium_data=np.array(calcium_data);
    calcium_data=calcium_data.T;

    calcium_data = calcium_data+np.random.normal(0, 1, calcium_data.shape);


    #plt.figure();
    mean_calcium =[];
    k=0;
    for ee in  indices_ii:
        #plt.plot(np.mean(calcium_data[ee[0]:ee[1],:],axis=0) -0.4*k, label=f'Neuron {i}')
        
        mm=np.mean(calcium_data[ee,:],axis=0);
        mmm=(mm-np.mean(mm))/np.std(mm);
        mean_calcium.append(mmm);
        #plt.plot(mmm -2.0*k,'k');
        
        k+=1;
        
        
    fc_vectors,fc_mat,sim_fc_mat=gso.fun_fc_vector(np.array(mean_calcium), 10,5); 
        

    exp_data=f'./subject_12_simulation_calcium_data_empirical_data.pkl';

    with open(exp_data, 'rb') as f:
            loaded_data = pickle.load(f)
                
    exp_fc_mat=loaded_data['whole_fc'];

    flat_matrix1 = np.abs(sim_fc_mat.flatten())
    flat_matrix2 = np.abs(exp_fc_mat.flatten())

    ii=flat_matrix1>0
    x_data=flat_matrix1[ii];
    y_data=flat_matrix2[ii];
    
    r_value=0;
    for i in range(len(x_data)):
        
        r_value+=np.abs(x_data[i]-y_data[i]);
        
    #slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

    print(xx, r_value)
    return r_value;
     

x0 = [0.1, 20];
bounds = [(0.01, 1), (10, 40)];
space = [
    Real(0.001, 10.0, name='log_std'),
    Real(10.0, 50.0, name='g_weight')
]

result = gp_minimize(fun_sc_weight, space, n_calls=30, random_state=42)

print("Best log_std:", result.x[0])
print("Best g_weight:", result.x[1])
print("Best score:", result.fun)
#0.43, 38.86
aaaa;

result = minimize(fun_sc_weight, x0,bounds=bounds)

# 결과 출력
x1_opt, x2_opt = result.x;
print(f"최적 x1: {x1_opt:.4f}, x2: {x2_opt:.4f}")

y_max = fun_sc_weight(x1_opt, x2_opt)

print(f"최적 x1: {x1_opt:.4f}, x2: {x2_opt:.4f}")
print(f"최대 y: {y_max:.4f}")
