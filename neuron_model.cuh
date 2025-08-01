#ifndef NEURON_MODEL_H
#define NEURON_MODEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <iostream> // For printf in C++
#include <time.h>   // For time(NULL)
#include <boost/python.hpp>


// Error checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// --- Structures ---

struct s_neuron_connection {
    std::vector<double> weight;
    std::vector<size_t> s_pre_id;
    std::vector<int> distance;
    
    void remove_duplicates() {
        std::unordered_set<size_t> seen;
        std::vector<size_t> new_s_pre_id;
        std::vector<double> new_weight;
        std::vector<int> new_distance;
        new_s_pre_id.reserve(s_pre_id.size());
        new_weight.reserve(weight.size());
        new_distance.reserve(distance.size());
        
        for (size_t i = 0; i < s_pre_id.size(); ++i)
        {
            if (seen.insert(s_pre_id[i]).second)
            { //second is true if insertion occurred (not duplicate)
                new_s_pre_id.push_back(s_pre_id[i]);
                new_weight.push_back(weight[i]);
                new_distance.push_back(distance[i]);
            }
        }
        
        s_pre_id.swap(new_s_pre_id);
        weight.swap(new_weight);
        distance.swap(new_distance);
    }
};

struct s_izkevich {
    double v = -70, u = 0;
    double c1 = 0.04, c2 = 5, c3 = 140, c4 = 1, c5 = -1, c6 = -3;
    double E_exc = 0.0, E_inh = 0.0;
    double a = 0.02, b = 0.2, c = -65, d = 8;
    double thre = 30;
    double exc_synap = 0, inh_synap = 0;
    double gamma_exc = 0.2, gamma_inh = 0.1;
    bool spike_checking = false, check_inh = false;
    double calcium = 0, spiking_time = -10000.0;
    double external_input = 0, noise_intensity = 15.0, noise_val = 0;
    double qe = 1, qi = 1;

    
};

// Struct to hold spike event data for output
struct SpikeEvent {
    int neuron_id;
    int time_step;
};




// --- Wrapper Class for Python ---
class NeuronSimulator {
public:
    // Changed constructor signature: removed connection_prob, added connections_list
     
    // 
    // NeuronSimulator(size_t num_neurons, double inh_prob, const std::vector<std::vector<size_t>>& connections_list, double g_weight,int final_time_steps, double dt_val, int ca_window);
   
    NeuronSimulator(size_t num_neurons, double noise_intensity, const std::vector<std::vector<int>>& distance_list,const std::vector<std::vector<int>>& connections_list, double inh_prob,  const std::vector<std::vector<double>>  &g_weight, int final_time_steps, double dt_val, int ca_window);

    ~NeuronSimulator();

    void run_simulation();

    // Methods to retrieve data

    boost::python::list get_calcium_data();
    boost::python::list get_spike_events();

private:
    size_t neuron_num;
    int final_time;
    double dt;
    int calcium_window;
    int ca_length;
    size_t max_possible_spikes;

    // Device pointers
    s_izkevich* d_izkevich = nullptr;
    int* d_nn = nullptr;
    int* d_pred_id = nullptr;
    int* d_post_id = nullptr;
    int* d_distance = nullptr;
    double* d_weight = nullptr;
    curandState* d_rand_states = nullptr;
    double* d_ca_data = nullptr;
    SpikeEvent* d_spike_records = nullptr;
    int* d_spike_count = nullptr;

    // Host-side mirrors (for initialization/results transfer)
    std::vector<s_izkevich> h_v_neuron;
    std::vector<s_neuron_connection> h_v_connection;
    std::vector<int> h_v_nn;
    std::vector<int> h_v_pred_id;
    std::vector<int> h_v_post_id;
    
    std::vector<int> h_v_distance;

    std::vector<double> h_v_weight;
};

#endif // NEURON_MODEL_H

