#include <boost/python.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <boost/math/distributions/students_t.hpp> // For Student's t-distribution
#include <Eigen/Dense>
#include <unordered_set>
#include <thread>
#include <future>
#include <numeric>

namespace bm = boost::math;
namespace p = boost::python;

Eigen::MatrixXd calculateCovariance(const Eigen::MatrixXd& data) {
    int num_rows = (int)data.rows();
    if (num_rows < 2) {
        // Throw a Python exception using boost::python::throw_error
        PyErr_SetString(PyExc_ValueError, "Error: Not enough observations to compute covariance (need at least 2).");
        p::throw_error_already_set();
        return Eigen::MatrixXd::Zero(data.cols(), data.cols()); // This line won't be reached
    }

    Eigen::RowVectorXd col_means = data.colwise().mean();
    Eigen::MatrixXd centered_data = data.rowwise() - col_means;

    return (centered_data.transpose() * centered_data) / (num_rows - 1);
}

// Function to convert a Python list of lists to Eigen::MatrixXd
Eigen::MatrixXd convertPyListToEigenMatrix(p::list py_list_of_lists) {
    if (p::len(py_list_of_lists) == 0) {
        return Eigen::MatrixXd::Zero(0, 0);
    }

    // Determine dimensions
    int num_rows = (int)p::len(py_list_of_lists);
    int num_cols = 0;

    // Assuming all inner lists have the same length
    if (num_rows > 0) {
        p::object first_row = py_list_of_lists[0];
        // Check if the first_row is actually a list
        if (PyList_Check(first_row.ptr())) {
            num_cols = (int)p::len(first_row);
        }
        else {
            // Handle case where it might be a flat list of numbers (1D array equivalent)
            // Or throw an error if nested lists are strictly required
            PyErr_SetString(PyExc_TypeError, "Input data must be a list of lists.");
            p::throw_error_already_set();
            return Eigen::MatrixXd::Zero(0, 0);
        }
    }

    if (num_cols == 0) { // Handle empty inner lists or single empty list
        return Eigen::MatrixXd::Zero(num_rows, 0);
    }

    Eigen::MatrixXd eigen_matrix(num_rows, num_cols);

    for (int i = 0; i < num_rows; ++i) {
        p::list row_list = p::extract<p::list>(py_list_of_lists[i]);
        if (p::len(row_list) != num_cols) {
            PyErr_SetString(PyExc_ValueError, "All inner lists must have the same length.");
            p::throw_error_already_set();
            return Eigen::MatrixXd::Zero(0, 0);
        }
        for (int j = 0; j < num_cols; ++j) {
            eigen_matrix(i, j) = p::extract<double>(row_list[j]);
        }
    }
    return eigen_matrix;
}

// Function to convert Eigen::MatrixXd to a Python list of lists
p::list convertEigenMatrixToPyList(const Eigen::MatrixXd& eigen_matrix) {
    p::list py_list_of_lists;
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        p::list row_list;
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            row_list.append(eigen_matrix(i, j));
        }
        py_list_of_lists.append(row_list);
    }
    return py_list_of_lists;
}


// Function to compute partial correlation-based functional connectivity (FC) matrix
// Takes Python list of lists as input, converts to Eigen::MatrixXd internally
// and returns Python list of lists.
p::list computePartialCorrelationFC_list_cpp(
    p::list py_data, // Python list of lists input
    bool zscore_normalize,
    const std::string& method)
{
    // Convert Python list of lists to Eigen::MatrixXd
    Eigen::MatrixXd data_eigen = convertPyListToEigenMatrix(py_data);

    if (data_eigen.rows() == 0 || data_eigen.cols() == 0) {
        // Return empty list if input was empty or invalid
        return p::list();
    }

    Eigen::MatrixXd processed_data = data_eigen;

    // 1. Z-score normalization
    if (zscore_normalize) {
        for (int c = 0; c < processed_data.cols(); ++c) {
            Eigen::VectorXd col = processed_data.col(c);
            double mean = col.mean();
            double std_dev = std::sqrt((col.array() - mean).square().sum() / (col.size() - 1)); // Sample std dev

            if (std_dev < 1e-9) { // Avoid division by zero for constant columns
                processed_data.col(c).setZero();
            }
            else {
                processed_data.col(c) = (col.array() - mean) / std_dev;
            }
        }
    }

    int n_regions = (int)processed_data.cols();

    Eigen::MatrixXd precision_matrix;

    if (method == "inverse_cov") {
        Eigen::MatrixXd cov_matrix = calculateCovariance(processed_data);

        if (cov_matrix.rows() == 0 || cov_matrix.cols() == 0) {
            PyErr_SetString(PyExc_ValueError, "Covariance matrix is empty after processing.");
            p::throw_error_already_set();
            return p::list();
        }

        Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);
        if (llt.info() != Eigen::Success) {
            PyErr_SetString(PyExc_ValueError, "Covariance matrix is singular or not positive definite for 'inverse_cov'. Cannot compute inverse.");
            p::throw_error_already_set();
            return p::list();
        }
        precision_matrix = llt.solve(Eigen::MatrixXd::Identity(n_regions, n_regions));

    }
    else if (method == "glasso") {
        PyErr_SetString(PyExc_ValueError, "Graphical Lasso method is selected but not fully implemented in this C++ example. You need to integrate a C++ GLasso library or implement it yourself.");
        p::throw_error_already_set();
        return p::list();

    }
    else {
        PyErr_SetString(PyExc_ValueError, "Invalid method. Must be 'glasso' or 'inverse_cov'.");
        p::throw_error_already_set();
        return p::list();
    }

    // 2. Partial correlation calculation from precision matrix
    Eigen::MatrixXd partial_corr_eigen = Eigen::MatrixXd::Zero(n_regions, n_regions);
    Eigen::VectorXd d = precision_matrix.diagonal().array().sqrt();

    for (int i = 0; i < n_regions; ++i) {
        for (int j = 0; j < n_regions; ++j) {
            if (i == j) {
                partial_corr_eigen(i, j) = 1.0;
            }
            else {
                if (d(i) > 1e-9 && d(j) > 1e-9) {
                    partial_corr_eigen(i, j) = -precision_matrix(i, j) / (d(i) * d(j));
                }
                else {
                    partial_corr_eigen(i, j) = 0.0;
                }
            }
        }
    }

    // Convert Eigen::MatrixXd back to a Python list of lists
    return convertEigenMatrixToPyList(partial_corr_eigen);
}



// --- Helper Functions (re-used) ---

// Function to calculate pairwise Euclidean distances
std::vector<std::vector<double>> pairwise_distances(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty()) {
        return {};
    }

    std::vector<std::vector<double>> D(A.size(), std::vector<double>(B.size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B.size(); ++j) {
            double sum_sq_diff = 0.0;
            size_t min_dim = std::min(A[i].size(), B[j].size());
            for (size_t k = 0; k < min_dim; ++k) {
                sum_sq_diff += (A[i][k] - B[j][k]) * (A[i][k] - B[j][k]);
            }
            D[i][j] = std::sqrt(sum_sq_diff);
        }
    }
    return D;
}

// --- Boost.Python Converters for Python List <-> C++ Vector ---

// Python list of lists (double) to std::vector<std::vector<double>>
struct DoubleVector2dFromList {
    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;

        Py_ssize_t outer_size = PyList_Size(obj_ptr);
        if (outer_size == 0) return Py_None;

        for (Py_ssize_t i = 0; i < outer_size; ++i) {
            PyObject* inner_list_ptr = PyList_GetItem(obj_ptr, i);
            if (!PyList_Check(inner_list_ptr)) return 0;

            Py_ssize_t inner_size = PyList_Size(inner_list_ptr);
            for (Py_ssize_t j = 0; j < inner_size; ++j) {
                PyObject* item_ptr = PyList_GetItem(inner_list_ptr, j);
                if (!PyFloat_Check(item_ptr) && !PyLong_Check(item_ptr)) {
                    if (PyObject_HasAttrString(item_ptr, "__float__")) {
                        continue;
                    }
                    return 0;
                }
            }
        }
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, p::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((p::converter::rvalue_from_python_storage<std::vector<std::vector<double>>>*)data)->storage.bytes;
        std::vector<std::vector<double>>* vec = new (storage) std::vector<std::vector<double>>();

        Py_ssize_t outer_size = PyList_Size(obj_ptr);
        vec->reserve(outer_size);

        for (Py_ssize_t i = 0; i < outer_size; ++i) {
            PyObject* inner_list_ptr = PyList_GetItem(obj_ptr, i);
            Py_ssize_t inner_size = PyList_Size(inner_list_ptr);
            std::vector<double> inner_vec;
            inner_vec.reserve(inner_size);

            for (Py_ssize_t j = 0; j < inner_size; ++j) {
                PyObject* item_ptr = PyList_GetItem(inner_list_ptr, j);
                inner_vec.push_back(p::extract<double>(item_ptr));
            }
            vec->push_back(inner_vec);
        }
        data->convertible = storage;
    }
};

// Python list of ints to std::vector<int>
struct IntVector1dFromList {
    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_Check(obj_ptr)) return 0;

        Py_ssize_t size = PyList_Size(obj_ptr);
        if (size == 0) return Py_None;

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item_ptr = PyList_GetItem(obj_ptr, i);
            if (!PyLong_Check(item_ptr)) {
                if (PyObject_HasAttrString(item_ptr, "__int__")) {
                    continue;
                }
                return 0;
            }
        }
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, p::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((p::converter::rvalue_from_python_storage<std::vector<int>>*)data)->storage.bytes;
        std::vector<int>* vec = new (storage) std::vector<int>();

        Py_ssize_t size = PyList_Size(obj_ptr);
        vec->reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item_ptr = PyList_GetItem(obj_ptr, i);
            vec->push_back(p::extract<int>(item_ptr));
        }
        data->convertible = storage;
    }
};


// std::vector<int> to Python list (for sel_neuron output)
struct IntVectorToList {
    static PyObject* convert(const std::vector<int>& vec) {
        p::list py_list;
        for (int x : vec) {
            py_list.append(x);
        }
        return p::incref(py_list.ptr());
    }
};

// std::vector<std::vector<double>> to Python list of lists (double)
// This converter was missing or incorrectly merged, defining it clearly here.
struct IntVector2dToList {
    static PyObject* convert(const std::vector<std::vector<int>>& vec) {
        p::list py_outer_list;
        for (const auto& inner_vec : vec) {
            p::list py_inner_list;
            for (int val : inner_vec) {
                py_inner_list.append(val);
            }
            py_outer_list.append(py_inner_list);
        }
        return p::incref(py_outer_list.ptr());
    }
};


struct DoubleVector2dToList {
    static PyObject* convert(const std::vector<std::vector<double>>& vec) {
        p::list py_outer_list;
        for (const auto& inner_vec : vec) {
            p::list py_inner_list;
            for (double val : inner_vec) {
                py_inner_list.append(val);
            }
            py_outer_list.append(py_inner_list);
        }
        return p::incref(py_outer_list.ptr());
    }
};

// NEW: std::vector<std::vector<bool>> to Python list of lists (bool)
struct BoolVector2dToList {
    static PyObject* convert(const std::vector<std::vector<bool>>& vec) {
        p::list py_outer_list;
        for (const auto& inner_vec : vec) {
            p::list py_inner_list;
            for (bool val : inner_vec) {
                py_inner_list.append(val);
            }
            py_outer_list.append(py_inner_list);
        }
        return p::incref(py_outer_list.ptr());
    }
};

p::tuple find_neuron_id_in_synapsee(
    const std::vector<std::vector<double>>& cell_data_xyz,
    const std::vector<std::vector<double>>& all_syn,double dd
    )
{

    std::vector<std::vector<double>> D = pairwise_distances(all_syn, cell_data_xyz);
    std::vector<std::vector<int>> neuron_id_in_synapse(all_syn.size());
    std::set<int> set_sel_synapse_id;
    
    for (size_t i = 0; i < D.size(); ++i)
    { // Iterating through all_syn rows
            for (size_t j = 0; j < D[i].size(); ++j) 
            { // Iterating through cell_data_xyz columns
                if (D[i][j] < dd) 
                {                   
                    set_sel_synapse_id.insert(int(i));
                    neuron_id_in_synapse[i].push_back(int(j));
                }
            }
    }
    
    std::vector<int> vec_sel_synapse_id(set_sel_synapse_id.begin(), set_sel_synapse_id.end());
    std::sort(vec_sel_synapse_id.begin(), vec_sel_synapse_id.end());
    std::vector<std::vector<int>> new_neuron_id_in_synapse(vec_sel_synapse_id.size());

    for (size_t i = 0; i < vec_sel_synapse_id.size(); i++) 
    {
        new_neuron_id_in_synapse[i] = neuron_id_in_synapse[vec_sel_synapse_id[i]];
    }

    
    return p::make_tuple(vec_sel_synapse_id, new_neuron_id_in_synapse);

}

p::tuple calculate_sc_from_synapse(
    const std::vector<std::vector<double>>& cell_data_xyz,
    const std::vector<std::vector<double>>& syn_xyz,
    const std::vector<int>& syn_idx
) {
    constexpr double dist_thr = 64.0;
    size_t syn_nn = syn_xyz.size();
    size_t cell_nn = cell_data_xyz.size();

    std::vector<std::vector<int>> neuron_idx_in_synapse(syn_nn);
    std::vector<std::vector<int>> syn_idx_in_neuron(cell_nn);
    std::vector<bool> sel_neuron(cell_nn, false);

    // 병렬 거리 계산 및 필터링 (std::async 사용)
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < syn_nn; i++) {
        futures.push_back(std::async(std::launch::async, [&syn_xyz, &cell_data_xyz, &neuron_idx_in_synapse, &sel_neuron, i] {
            const auto& syn = syn_xyz[i];
            for (size_t j = 0; j < cell_data_xyz.size(); j++) {
                const auto& cell = cell_data_xyz[j];

                double dx = syn[0] - cell[0];
                double dy = syn[1] - cell[1];
                double dz = syn[2] - cell[2];

                if (dx * dx + dy * dy + dz * dz < dist_thr) {
                    neuron_idx_in_synapse[i].push_back(int(j));
                    sel_neuron[j] = true;
                }
            }
            }));
    }
    for (auto& fut : futures) fut.get(); // 모든 병렬 작업 완료 대기
    
    // 신경세포-시냅스 매핑 (병렬 실행)
    futures.clear();
    for (size_t i = 0; i < syn_nn; i++) {
        for (int j : neuron_idx_in_synapse[i]) {
            syn_idx_in_neuron[j].push_back(syn_idx[i]);
        }
    }
    
   
    for (auto& vec : syn_idx_in_neuron) {
        futures.push_back(std::async(std::launch::async, [&vec] {
            std::sort(vec.begin(), vec.end());
            }));
    }
    for (auto& fut : futures) fut.get();

    // 선택된 뉴런 추출
    std::vector<int> sel_neuron_vec;
    for (size_t i = 0; i < cell_nn; i++) {
        if (sel_neuron[i]) sel_neuron_vec.push_back(int(i));
    }

    size_t nn_sc_mat = sel_neuron_vec.size();
    std::vector<std::vector<int>> sc_mat(nn_sc_mat, std::vector<int>(nn_sc_mat, 0));

    // 중복 요소 확인을 위한 함수
    auto hasCommonElement = [](const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> common_elements;
        common_elements.reserve(std::min(a.size(), b.size()));
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(common_elements));
        return !common_elements.empty();
        };

    // 신경세포 연결 행렬 생성 (병렬 처리)
    futures.clear();
    for (size_t i = 0; i < nn_sc_mat; i++) {
        futures.push_back(std::async(std::launch::async, [&sc_mat, &sel_neuron_vec, &syn_idx_in_neuron, &hasCommonElement,i] {
            for (size_t j = i + 1; j < sel_neuron_vec.size(); j++) {
                if (hasCommonElement(syn_idx_in_neuron[sel_neuron_vec[i]], syn_idx_in_neuron[sel_neuron_vec[j]])) {
                    sc_mat[i][j] = 1;
                    sc_mat[j][i] = 1;
                }
            }
            }));
    }

    for (auto& fut : futures) fut.get();

    return p::make_tuple(sel_neuron_vec, sc_mat);
}





/*
p::tuple calculate_sc_from_synapse(
    const std::vector<std::vector<double>>& cell_data_xyz,
    const std::vector<std::vector<double>>& syn_xyz,
    const std::vector<int>& syn_idx
) {
    // Stage 1: Initial Distance Calculation and
    // Filtering
    constexpr double dist_thr = 64.0;  // 8 * 8 값 미리 계산
    size_t syn_nn = syn_xyz.size();
    size_t cell_nn = cell_data_xyz.size();

    std::vector<std::vector<int>> neuron_idx_in_synapse(syn_nn);
    std::vector<std::vector<int>> syn_idx_in_neuron(cell_nn);
    std::vector<bool> sel_neuron(cell_nn, false);

    // 거리 계산 및 필터링
    for (size_t i = 0; i < syn_nn; i++) {
        const auto& syn = syn_xyz[i];  // 참조 사용
        for (size_t j = 0; j < cell_nn; j++) {
            const auto& cell = cell_data_xyz[j];

            double dx = syn[0] - cell[0];
            double dy = syn[1] - cell[1];
            double dz = syn[2] - cell[2];

            if (dx * dx + dy * dy + dz * dz < dist_thr) {
                neuron_idx_in_synapse[i].push_back(j);
                sel_neuron[j] = true;
            }
        }
    }

    // 신경세포-시냅스 매핑
    for (size_t i = 0; i < syn_nn; i++) {
        for (int j : neuron_idx_in_synapse[i]) {
            syn_idx_in_neuron[j].push_back(syn_idx[i]);
        }
    }

    // 정렬 수행
    for (auto& vec : syn_idx_in_neuron) {
        std::sort(vec.begin(), vec.end());
    }

    // 선택된 뉴런만 추출
    std::vector<int> sel_neuron_vec;
    for (size_t i = 0; i < cell_nn; i++) {
        if (sel_neuron[i]) sel_neuron_vec.push_back(i);
    }

    size_t nn_sc_mat = sel_neuron_vec.size();
    std::vector<std::vector<int>> sc_mat(nn_sc_mat, std::vector<int>(nn_sc_mat, 0));

    // 중복 요소 확인을 위한 람다 함수
    auto hasCommonElement = [](const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> common_elements;
        // Reserve some memory to avoid reallocations if many common elements are expected.
        // A rough estimate could be std::min(a.size(), b.size()).
        common_elements.reserve(std::min(a.size(), b.size()));

        std::set_intersection(a.begin(), a.end(),
            b.begin(), b.end(),
            std::back_inserter(common_elements));
        return !common_elements.empty();
        };


  
    // 신경세포 연결 행렬 생성
    for (size_t i = 0; i < nn_sc_mat; i++) {
        for (size_t j = i+1; j < nn_sc_mat; j++) {
            if (hasCommonElement(syn_idx_in_neuron[sel_neuron_vec[i]], syn_idx_in_neuron[sel_neuron_vec[j]])) {
                sc_mat[i][j] = 1;
                sc_mat[j][i] = 1;

            }
        }
    }
    
                

    // Return the tuple of results
    return p::make_tuple(sel_neuron_vec, sc_mat);
}
*/


//const std::vector<int>& cell_data_id,
//const std::vector<std::vector<double>>& spot_data, // Still passed, but not used in this specific calculation.
// --- Main Neuron Selection Logic Function (Optimized from previous iteration) ---
p::tuple calculate_sel_neuron_sc_distance(
    const std::vector<std::vector<double>>& cell_data_xyz,
    const std::vector<std::vector<double>>& all_syn,
    const std::vector<int>& syn_idx
) {
    // Stage 1: Initial Distance Calculation and
    // Filtering
    std::vector<std::vector<double>> D = pairwise_distances(all_syn, cell_data_xyz);

    // Intermediate structure: map from original cell_data_xyz index to a set of associated original syn_ids
    std::map<int, std::set<int>> temp_neuron_to_syn_id_map;
    std::set<int> i_neuron_indices_set; // This will become sel_neuron indices

    for (size_t i = 0; i < D.size(); ++i) { // Iterating through all_syn rows
        for (size_t j = 0; j < D[i].size(); ++j) { // Iterating through cell_data_xyz columns
            if (D[i][j] < 8) {
                if (i < syn_idx.size()) { // Ensure syn_idx is valid
                    int original_syn_id_val = syn_idx[i]; // This is the value from the 4th column of Python's synapse_xyz
                    temp_neuron_to_syn_id_map[static_cast<int>(j)].insert(original_syn_id_val);
                    i_neuron_indices_set.insert(static_cast<int>(j)); // Collect unique neuron indices
                }
            }
        }
    }

    std::vector<int> i_neuron_indices(i_neuron_indices_set.begin(), i_neuron_indices_set.end());
    std::sort(i_neuron_indices.begin(), i_neuron_indices.end()); // This is your sel_neuron

    // Construct new_neuron_xyz (pos) based on sel_neuron
    std::vector<std::vector<double>> new_neuron_xyz_pos;
    new_neuron_xyz_pos.reserve(i_neuron_indices.size());
    for (int neuron_idx : i_neuron_indices) {
        if (neuron_idx >= 0 && neuron_idx < cell_data_xyz.size()) {
            new_neuron_xyz_pos.push_back(cell_data_xyz[neuron_idx]);
        }
    }

    // Convert the map of sets to vector of sets for direct usage in sc_mat calculation
    std::vector<std::set<int>> neuron_to_syn_id_sets(new_neuron_xyz_pos.size());
    std::map<int, int> original_neuron_idx_to_new_idx;
    for (size_t i = 0; i < i_neuron_indices.size(); ++i) {
        original_neuron_idx_to_new_idx[i_neuron_indices[i]] = static_cast<int>(i);
    }

    for (const auto& pair : temp_neuron_to_syn_id_map) {
        int original_neuron_idx = pair.first;
        if (original_neuron_idx_to_new_idx.count(original_neuron_idx)) {
            int new_idx = original_neuron_idx_to_new_idx[original_neuron_idx];
            neuron_to_syn_id_sets[new_idx] = pair.second; // Copy the set directly
        }
    }


    // --- D3 (Distance Matrix) Calculation ---
    // Calculate squared distances to avoid repeated sqrt if only relative distances matter
    // Or if D3 is truly Euclidean, stick to current pairwise_distances which calculates sqrt.
    // Assuming D3 is for plotting or further spatial analysis, leave as is.
    std::vector<std::vector<double>> D3 = pairwise_distances(new_neuron_xyz_pos, new_neuron_xyz_pos);


    // --- SC Matrix Calculation (Optimized using pre-built sets) ---
    size_t n = new_neuron_xyz_pos.size(); // n is the size of sel_neuron
    std::vector<std::vector<double>> sc_mat(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const std::set<int>& set1 = neuron_to_syn_id_sets[i];
            const std::set<int>& set2 = neuron_to_syn_id_sets[j];

            bool intersection_found = false;
            // Iterate through smaller set and check in larger set for efficiency
            if (set1.size() < set2.size()) {
                for (int val : set1) {
                    if (set2.count(val)) {
                        intersection_found = true;
                        break;
                    }
                }
            }
            else {
                for (int val : set2) {
                    if (set1.count(val)) {
                        intersection_found = true;
                        break;
                    }
                }
            }

            if (intersection_found) {
                sc_mat[i][j] = 1.0;
                sc_mat[j][i] = 1.0;
            }
        }
    }

    // Return the tuple of results
    return p::make_tuple(i_neuron_indices, sc_mat, D3);
}

// --- fun_fdr Implementation in C++ (Optimized) ---
std::vector<std::vector<bool>> fun_fdr_cpp(double q, int nT, const std::vector<std::vector<double>>& fc) {
    if (fc.empty()) {
        return {};
    }

    size_t N = fc.size();
    if (N == 0 || (N > 0 && fc[0].size() != N)) {
        PyErr_SetString(PyExc_ValueError, "Input 'fc' must be a square matrix (N x N).");
        p::throw_error_already_set();
    }

    // Initialize pval_mat with 1.0
    std::vector<std::vector<double>> pval_mat(N, std::vector<double>(N, 1.0));

    double df = static_cast<double>(nT - 2);

    // Create Boost.Math Student's t-distribution object once
    // Handle cases where df is invalid for the distribution
    if (df <= 0) {
        // If degrees of freedom are invalid, all p-values are considered 1.0
        // and no significance is found, so return all false.
        return std::vector<std::vector<bool>>(N, std::vector<bool>(N, false));
    }
    bm::students_t dist(df);

    // Calculate p-values for upper triangle
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double r = fc[i][j];
            double t_stat_abs = std::abs(r);

            double denominator = (1.0 - r * r);
            // Handle edge case where r^2 is close to 1, causing division by zero or negative
            if (denominator <= 0) {
                pval_mat[i][j] = pval_mat[j][i] = 1.0; // Invalid t-stat, assume no significance
                continue;
            }

            double t_stat = t_stat_abs * std::sqrt(df / denominator);

            // Calculate two-tailed p-value using the pre-created distribution object
            double pval = 2 * (1.0 - bm::cdf(dist, t_stat));

            pval_mat[i][j] = pval;
            pval_mat[j][i] = pval; // Symmetrical
        }
    }

    // Extract p-values from lower triangle (excluding diagonal)
    std::vector<double> pvals;
    pvals.reserve(N * (N - 1) / 2);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) { // Lower triangle
            pvals.push_back(pval_mat[i][j]);
        }
    }

    if (pvals.empty()) {
        return std::vector<std::vector<bool>>(N, std::vector<bool>(N, false));
    }

    // Sort p-values (and their original indices)
    std::vector<size_t> sorted_idx(pvals.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);

    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](size_t a, size_t b) {
        return pvals[a] < pvals[b];
        });

    std::vector<double> sorted_p_values(pvals.size());
    for (size_t i = 0; i < pvals.size(); ++i) {
        sorted_p_values[i] = pvals[sorted_idx[i]];
    }

    size_t m = pvals.size();
    double max_cutoff_p = 0.0;
    bool found_significant = false;

    // Apply FDR correction
    for (size_t k = 0; k < m; ++k) { // k is 0-based
        double threshold = (static_cast<double>(k + 1) / m) * q;
        if (sorted_p_values[k] <= threshold) {
            max_cutoff_p = sorted_p_values[k];
            found_significant = true;
        }
    }

    std::vector<std::vector<bool>> sig_mat(N, std::vector<bool>(N, false));

    if (found_significant) {
        // Apply the cutoff to the original pval_mat
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sig_mat[i][j] = (pval_mat[i][j] <= max_cutoff_p);
            }
        }
    }

    return sig_mat;
}


// --- Boost.Python Module Definition ---
BOOST_PYTHON_MODULE(analysis_assitance_python) {
    // Register custom converters
    p::converter::registry::push_back(&DoubleVector2dFromList::convertible,
        &DoubleVector2dFromList::construct,
        p::type_id<std::vector<std::vector<double>>>());

    p::converter::registry::push_back(&IntVector1dFromList::convertible,
        &IntVector1dFromList::construct,
        p::type_id<std::vector<int>>());

    // Register output converters
    p::to_python_converter<std::vector<int>, IntVectorToList>();
    p::to_python_converter<std::vector<std::vector<double>>, DoubleVector2dToList>(); // For sc_mat and D3
    p::to_python_converter<std::vector<std::vector<bool>>, BoolVector2dToList>();     // For fun_fdr_cpp output
    p::to_python_converter<std::vector<std::vector<int>>, IntVector2dToList>();
    // Expose the main function to Python


    p::def("find_neuron_id_in_synapsee", &find_neuron_id_in_synapsee,
            (p::arg("cell_data_xyz"), p::arg("all_syn")),
            "Calculates selected neuron indices (sel_neuron) and related matrices from various input lists.");

    p::def("calculate_sel_neuron_sc_distance", &calculate_sel_neuron_sc_distance,
        (p::arg("cell_data_xyz"),  p::arg("all_syn"), p::arg("syn_idx")),
        "Calculates selected neuron indices (sel_neuron) and related matrices from various input lists.");

    p::def("fun_fdr", &fun_fdr_cpp,
        (p::arg("q"), p::arg("nT"), p::arg("fc")),
        "Performs False Discovery Rate (FDR) correction on a correlation matrix.");

    p::def("compute_partial_correlation_fc", &computePartialCorrelationFC_list_cpp,
        (p::arg("data"), p::arg("zscore_normalize") = true, p::arg("method") = "inverse_cov"),
        "Compute partial correlation-based functional connectivity (FC) matrix from Python lists.");

    p::def("calculate_sc_from_synapse", &calculate_sc_from_synapse,
        (p::arg("cell_data_xyz"), p::arg("syn_xyz") = true, p::arg("syn_idx")),
        "Compute partial correlation-based functional connectivity (FC) matrix from Python lists.");

}

