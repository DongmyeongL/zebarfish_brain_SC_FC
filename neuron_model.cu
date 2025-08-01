#include "neuron_model.cuh" // Include the header with definitions
#include <time.h>         // For time(NULL)
#include <algorithm>      // For std::min

// --- NeuronSimulator Class Implementation ---
// --- CUDA Kernels ---
__global__ void setup_rand(curandState* __restrict__ state, int seed, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) curand_init(seed + tid, tid, 0, &state[tid]);
}

__global__ void cuda_fun_connect_update_by_link(int current_time, s_izkevich* __restrict__ v_neuron,
    const int* __restrict__ p_post_id,
    const int* __restrict__ p_pre_id,
    const int* __restrict__ p_distance,
    const double* __restrict__ p_weight,
    int size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    

    int pre_id = p_pre_id[tid];

    int post_id = p_post_id[tid];
    
    int delay = p_distance[tid];

    int pre_spiking_time = v_neuron[pre_id].spiking_time;

    //if(v_neuron[pre_id].spike_checking)
    if ( current_time==pre_spiking_time+delay*1)
    {
        
        if (v_neuron[pre_id].check_inh)
        {       
                atomicAdd(&v_neuron[post_id].E_inh, p_weight[tid]);
        }
        else
        {
                atomicAdd(&v_neuron[post_id].E_exc, p_weight[tid]);
        }
    }

    
}



__global__ void cuda_fun_connect_update(s_izkevich* __restrict__ v_neuron,
    const int* __restrict__ p_nn,
    const int* __restrict__ p_pre_id,
    const double* __restrict__ p_weight,
    int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid >= size) return;

    int st = (tid == 0) ? 0 : p_nn[tid - 1];
    int ft = p_nn[tid];

    double E_exc_sum = 0.0;
    double E_inh_sum = 0.0;
    for (int j = st; j < ft; j++) {
        int kid = p_pre_id[j];
        if (v_neuron[kid].spike_checking) {
            if (v_neuron[kid].check_inh)
                E_inh_sum += p_weight[j];
            else
                E_exc_sum += p_weight[j];
        }
    }
    v_neuron[tid].E_exc = E_exc_sum; // Assign the calculated sum
    v_neuron[tid].E_inh = E_inh_sum; // Assign the calculated sum
}


__global__ void cuda_fun_update_ca_opt(s_izkevich* __restrict__ v_neuron,
    curandState* __restrict__ state,
    double dt_sim,
    int time_step,
    int size,
    SpikeEvent* __restrict__ d_spike_records,
    int* __restrict__ d_spike_count,
    int max_spikes_limit) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    // Load neuron state into registers (fast local memory)
    s_izkevich neuron = v_neuron[tid];

    // Synaptic dynamics
    neuron.exc_synap += dt_sim * (-neuron.gamma_exc * neuron.exc_synap + neuron.E_exc * neuron.qe);
    neuron.inh_synap += dt_sim * (-neuron.gamma_inh * neuron.inh_synap + neuron.E_inh * neuron.qi);

    // Reset accumulated synaptic inputs for the next time step *before* calculating v,u
    // This is important: E_exc/E_inh reflect inputs from the *previous* step.
    // This reset logic is crucial for the non-atomic update in cuda_fun_connect_update to work correctly.
    neuron.E_exc = 0.0;
    neuron.E_inh = 0.0;

    // Izhikevich neuron dynamics
    double tempu = neuron.a * (neuron.b * neuron.v - neuron.u);
    double tempv = neuron.c1 * neuron.v * neuron.v + neuron.c2 * neuron.v + neuron.c3 - neuron.c4 * neuron.u + neuron.c5 * neuron.exc_synap * (neuron.v - 10.0) + neuron.c6 * neuron.inh_synap * (neuron.v + 80);

    // Noise update (Ornstein-Uhlenbeck-like process)
    neuron.noise_val += dt_sim * (-0.1 * neuron.noise_val + neuron.noise_intensity * curand_normal_double(&state[tid]));

    // Update v and u
    neuron.v += dt_sim * (tempv + neuron.noise_val);
    neuron.u += dt_sim * tempu;

    // Calcium dynamics

    double ica = 1.0 / (1.0 + exp(-1.1 * (neuron.v + 34.0))); // Use 1.0 for double division
    neuron.calcium += dt_sim * (5.8 * ica - 0.008 * neuron.calcium);
    


    // Spike check and reset
    bool had_spike = false;
    if (neuron.v >= neuron.thre) {
        neuron.v = neuron.c;
        neuron.u += neuron.d;
        neuron.spike_checking = true; // Mark as spiking for this step
        neuron.spiking_time = time_step;
        had_spike = true;
        
        neuron.thre += 0.01*(30.0 - neuron.thre)*dt_sim+30;
    }
    else {
        neuron.spike_checking = false; // Not spiking this step
        neuron.thre += 0.01 * (30.0 - neuron.thre) * dt_sim ;
    }


     
    // Record spike event if neuron spiked AND there's space in the buffer
    if (had_spike) {
        int current_spike_idx = atomicAdd(d_spike_count, 1);
        if (current_spike_idx < max_spikes_limit) {
            d_spike_records[current_spike_idx].neuron_id = tid;
            d_spike_records[current_spike_idx].time_step = time_step;
        }
        else {
            // If buffer is full, decrement the counter back to prevent overflow
            atomicSub(d_spike_count, 1);
        }
    }

    
    // Write neuron state back to global memory
    v_neuron[tid] = neuron;
}

__global__ void cuda_fun_ca_save(const s_izkevich* __restrict__ v_neuron,
    double* __restrict__ p_ca_data,
    int current_time_idx, // Index for calcium_data array
    int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        p_ca_data[current_time_idx * size + tid] = v_neuron[tid].calcium;
}



// Changed constructor signature: removed connection_prob, added connections_list
//NeuronSimulator::NeuronSimulator(size_t num_neurons, double inh_prob, const std::vector<std::vector<size_t>>& connections_list, double g_weight,int final_time_steps, double dt_val, int ca_window)
NeuronSimulator::NeuronSimulator(size_t num_neurons, double noise_int, const std::vector<std::vector<int>>& distance_list,const std::vector<std::vector<int>>& connections_list, double inh_prob, const std::vector<std::vector<double>> &g_weight, int final_time_steps, double dt_val, int ca_window)
   : neuron_num(num_neurons),
    final_time(final_time_steps),
    dt(dt_val),
    calcium_window(ca_window)
{
    // Basic validation for connections_list
    //std::vector<std::vector<size_t>>  connections_list(num_neurons);

    if (connections_list.size() != neuron_num) {
        std::cerr << "Error: connections_list size (" << connections_list.size()
            << ") does not match num_neurons (" << neuron_num << ").\n";
        exit(EXIT_FAILURE); // Or throw an exception
    }
    
    // Calculate ca_length
    ca_length = final_time / calcium_window;
    if (final_time % calcium_window != 0) {
        ca_length++;
    }

    // Initialize host-side neuron data and connections
    h_v_neuron.resize(neuron_num);
    h_v_connection.resize(neuron_num);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Set inhibition neuron properties
    for (size_t i = 0; i < neuron_num; ++i) {
        h_v_neuron[i].noise_intensity = noise_int;
    }

    for (size_t i = 0; i < neuron_num; ++i) {
        if (dis(gen) < inh_prob) {
            h_v_neuron[i].check_inh = true;
            h_v_neuron[i].a = 0.1;
            h_v_neuron[i].d = 2;
        }
    }

    // --- Set connections based on provided connections_list ---
    size_t total_connections_count = 0;

    for (size_t i = 0; i < neuron_num; ++i) {
        // Clear existing connections if any (though h_v_connection is newly sized)
        h_v_connection[i].s_pre_id.clear();
        h_v_connection[i].weight.clear();
        h_v_connection[i].distance.clear();

        int k=0;
        for (size_t pre_id : connections_list[i]) {
            if (pre_id >= neuron_num) {
                std::cerr << "Error: Invalid pre-synaptic neuron ID " << pre_id
                    << " for post-synaptic neuron " << i << ". Max ID is " << neuron_num - 1 << ".\n";
                exit(EXIT_FAILURE);
            }
            h_v_connection[i].s_pre_id.push_back(pre_id);
            h_v_connection[i].weight.push_back(g_weight[i][k]); // Assign uniform weight
            h_v_connection[i].distance.push_back(distance_list[i][k]);
            k++;
        }
        h_v_connection[i].remove_duplicates(); // Remove duplicates (if any in input list)
        
        //normalzing;
        double norm_ll = (double)h_v_connection[i].s_pre_id.size();

        if (h_v_connection[i].s_pre_id.size() > 0)
        {   
            double t_norm = 0;
            for (int k = 0; k < h_v_connection[i].weight.size(); k++)
            {
                    t_norm += h_v_connection[i].weight[k];
            }
            for (int k = 0; k < h_v_connection[i].weight.size(); k++)
            {
                    h_v_connection[i].weight[k] = h_v_connection[i].weight[k]/t_norm*45;
                   
            }
        }


        total_connections_count += h_v_connection[i].s_pre_id.size();
    }

    // End of connection setting logic

    //

    // Prepare data for CUDA memory transfer (flattened connection data)
    h_v_pred_id.reserve(total_connections_count);
    h_v_weight.reserve(total_connections_count);
    h_v_post_id.reserve(total_connections_count);
    h_v_distance.reserve(total_connections_count);
    
    h_v_nn.resize(neuron_num);

    int current_offset = 0;
    for (size_t i = 0; i < neuron_num; ++i) {

        for (size_t j = 0 ; j <this->h_v_connection[i].s_pre_id.size() ; j++ )
        {
            h_v_post_id.push_back(i);
        }


        h_v_pred_id.insert(h_v_pred_id.end(), h_v_connection[i].s_pre_id.begin(), h_v_connection[i].s_pre_id.end());
        h_v_weight.insert(h_v_weight.end(), h_v_connection[i].weight.begin(), h_v_connection[i].weight.end());
        h_v_distance.insert(h_v_distance.end(), h_v_connection[i].distance.begin(), h_v_connection[i].distance.end());
        
        current_offset += h_v_connection[i].s_pre_id.size();
        h_v_nn[i] = current_offset;
    }

    // --- CUDA Memory Allocation (Unified Memory for simplicity) ---
    CUDA_CHECK(cudaMallocManaged((void**)&d_izkevich, neuron_num * sizeof(s_izkevich)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_nn, h_v_nn.size() * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_pred_id, h_v_pred_id.size() * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_post_id, h_v_post_id.size() * sizeof(int)));
    
    CUDA_CHECK(cudaMallocManaged((void**)&d_distance, h_v_distance.size() * sizeof(int)));

    CUDA_CHECK(cudaMallocManaged((void**)&d_weight, h_v_weight.size() * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_rand_states, neuron_num * sizeof(curandState)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_ca_data, neuron_num * ca_length * sizeof(double)));

    // Max spikes: Adjust this based on expected firing rate.
    max_possible_spikes = neuron_num * final_time *0.001;
    CUDA_CHECK(cudaMallocManaged((void**)&d_spike_records, max_possible_spikes * sizeof(SpikeEvent)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_spike_count, sizeof(int)));
    *d_spike_count = 0; // Initialize spike count to 0

    // Copy initial host data to Unified Memory
    CUDA_CHECK(cudaMemcpy(d_izkevich, h_v_neuron.data(), neuron_num * sizeof(s_izkevich), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nn, h_v_nn.data(), h_v_nn.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pred_id, h_v_pred_id.data(), h_v_pred_id.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_post_id, h_v_post_id.data(), h_v_post_id.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_weight, h_v_weight.data(), h_v_weight.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_distance, h_v_distance.data(), h_v_distance.size() * sizeof(int), cudaMemcpyHostToDevice));


    // Initialize cuRAND states once
    int threads_per_block = 256;
    int blocks_per_grid = (neuron_num + threads_per_block - 1) / threads_per_block;
    setup_rand << <blocks_per_grid, threads_per_block >> > (d_rand_states, (int)time(NULL), neuron_num);
    
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "NeuronSimulator initialized with " << neuron_num << " neurons for " << final_time << " steps." << std::endl;
}

NeuronSimulator::~NeuronSimulator() {
    CUDA_CHECK(cudaFree(d_izkevich));
    CUDA_CHECK(cudaFree(d_nn));
    CUDA_CHECK(cudaFree(d_pred_id));
    CUDA_CHECK(cudaFree(d_post_id));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_rand_states));
    CUDA_CHECK(cudaFree(d_ca_data));
    CUDA_CHECK(cudaFree(d_spike_records));
    CUDA_CHECK(cudaFree(d_spike_count));
    CUDA_CHECK(cudaFree(d_distance));

    std::cout << "NeuronSimulator resources freed." << std::endl;
}

void NeuronSimulator::run_simulation() {
    int threads_per_block = 256;
    int blocks_per_grid = (neuron_num + threads_per_block - 1) / threads_per_block;
    int ca_count = 0;
    bool simulation_aborted_due_to_spikes = false;


    int link_blocks_per_grid= (this->h_v_post_id.size() + threads_per_block - 1) / threads_per_block;
    std::cout << "Running simulation..." << std::endl;
    for (int t = 0; t < final_time; ++t) {

        // Connection Update Kernel

        //cuda_fun_connect_update << <blocks_per_grid, threads_per_block >> > (d_izkevich, d_nn, d_pred_id, d_weight, neuron_num); //csr
        cuda_fun_connect_update_by_link << <link_blocks_per_grid, threads_per_block >> > (t, d_izkevich, d_post_id, d_pred_id,d_distance, d_weight, this->h_v_post_id.size()); //coo

        CUDA_CHECK(cudaDeviceSynchronize());

        // Neuron State Update and Spike Recording Kernel
        cuda_fun_update_ca_opt << <blocks_per_grid, threads_per_block >> > (
            d_izkevich, d_rand_states, dt, t, neuron_num,
            d_spike_records, d_spike_count, max_possible_spikes);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check for spike buffer overflow on the host after the kernel
        if (t % 1000 == 0) {
            int current_spike_count;
            CUDA_CHECK(cudaMemcpy(&current_spike_count, d_spike_count, sizeof(int), cudaMemcpyDeviceToHost));

            if ((size_t)current_spike_count > max_possible_spikes) {
                std::cerr << "\nERROR: Spike record buffer overflowed at time step " << t << "!\n";
                std::cerr << "       Recorded " << current_spike_count << " spikes, but buffer limit is " << max_possible_spikes << ".\n";
                std::cerr << "       Please increase 'max_possible_spikes' in NeuronSimulator constructor and rerun the simulation.\n";
                simulation_aborted_due_to_spikes = true;
                break; // Abort the simulation loop
            }
        }
        // Calcium Data Save
        if (t % calcium_window == 0) {
            cuda_fun_ca_save << <blocks_per_grid, threads_per_block >> > (d_izkevich, d_ca_data, ca_count++, neuron_num);
          //  CUDA_CHECK(cudaDeviceSynchronize());
        }


    }

    std::cout << "Simulation finished." << std::endl;
    if (simulation_aborted_due_to_spikes) {
        std::cerr << "Simulation aborted due to spike buffer overflow.\n";
    }
}

// Directly convert to boost::python::list
boost::python::list NeuronSimulator::get_calcium_data() {
    boost::python::list py_list;
    // Copy data from device to host
    std::vector<double> h_ca_data(neuron_num * ca_length);
    CUDA_CHECK(cudaMemcpy(h_ca_data.data(), d_ca_data, neuron_num * ca_length * sizeof(double), cudaMemcpyDeviceToHost));

    // Append elements to Python list

    for (size_t i = 0; i <  ca_length; ++i) 
    {
        boost::python::list t_out_data;

        for (size_t j = 0; j < neuron_num; ++j)
        {
            t_out_data.append(h_ca_data[i * neuron_num + j]);
        }
        
        py_list.append(t_out_data);
    
    }
    return py_list;
}

boost::python::list NeuronSimulator::get_spike_events() {
    boost::python::list py_list;
    int total_spikes;
    CUDA_CHECK(cudaMemcpy(&total_spikes, d_spike_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Ensure we don't try to copy more than what was allocated
    total_spikes = std::min((size_t)total_spikes, max_possible_spikes);

    // Copy data from device to host
    std::vector<SpikeEvent> h_spike_records(total_spikes);
    if (total_spikes > 0) {
        CUDA_CHECK(cudaMemcpy(h_spike_records.data(), d_spike_records, total_spikes * sizeof(SpikeEvent), cudaMemcpyDeviceToHost));
    }

    // Append SpikeEvent as Python tuples to list
    for (int i = 0; i < total_spikes; ++i) {
        py_list.append(boost::python::make_tuple(h_spike_records[i].neuron_id, h_spike_records[i].time_step));
    }
    return py_list;
}


