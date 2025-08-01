#include <boost/python.hpp>
// #include <boost/python/suite/vector_indexing_suite.hpp> // <-- 이 헤더를 제거합니다.
#include "neuron_model.cuh" // Your C++ CUDA header

// Custom converter for std::vector<size_t> to Python list for input
struct vector_to_py_list_size_t {
    static PyObject* convert(const std::vector<size_t>& vec) {
        boost::python::list py_list;
        for (size_t val : vec) {
            py_list.append(val);
        }
        return boost::python::incref(py_list.ptr());
    }
};

// Custom converter for std::vector<std::vector<size_t>> to Python list of lists for input
struct vector_of_vectors_to_py_list_of_lists_size_t {
    static PyObject* convert(const std::vector<std::vector<size_t>>& vec_of_vec) {
        boost::python::list outer_list;
        for (const auto& inner_vec : vec_of_vec) {
            boost::python::list inner_list;
            for (size_t val : inner_vec) {
                inner_list.append(val);
            }
            outer_list.append(inner_list);
        }
        return boost::python::incref(outer_list.ptr());
    }
};
using namespace boost::python;


struct doubleVector2dFromList {

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
    static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((converter::rvalue_from_python_storage<std::vector<std::vector<double>>>*)data)->storage.bytes;
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
                inner_vec.push_back(extract<double>(item_ptr));
            }
            vec->push_back(inner_vec);
        }
        data->convertible = storage;
    }
};

struct intVector2dFromList {

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
    static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((converter::rvalue_from_python_storage<std::vector<std::vector<int>>>*)data)->storage.bytes;
        std::vector<std::vector<int>>* vec = new (storage) std::vector<std::vector<int>>();

        Py_ssize_t outer_size = PyList_Size(obj_ptr);
        vec->reserve(outer_size);

        for (Py_ssize_t i = 0; i < outer_size; ++i) {
            PyObject* inner_list_ptr = PyList_GetItem(obj_ptr, i);
            Py_ssize_t inner_size = PyList_Size(inner_list_ptr);
            std::vector<int> inner_vec;
            inner_vec.reserve(inner_size);

            for (Py_ssize_t j = 0; j < inner_size; ++j) {
                PyObject* item_ptr = PyList_GetItem(inner_list_ptr, j);
                inner_vec.push_back(extract<int>(item_ptr));
            }
            vec->push_back(inner_vec);
        }
        data->convertible = storage;
    }
};


BOOST_PYTHON_MODULE(MONET_SNN_CUDA_PYTHON_BOOST) { // Module name: monet_snn_cuda_python_boost

    using namespace boost::python;

    converter::registry::push_back(&intVector2dFromList::convertible,
        &intVector2dFromList::construct,
        type_id<std::vector<std::vector<int>>>());

    converter::registry::push_back(&doubleVector2dFromList::convertible,
        &doubleVector2dFromList::construct,
        type_id<std::vector<std::vector<double>>>());

    // SpikeEvent struct is no longer exposed as a class,
    // as it's converted to a tuple in get_spike_events()

    // Register converters for input parameters
    to_python_converter<std::vector<size_t>, vector_to_py_list_size_t>();
    to_python_converter<std::vector<std::vector<size_t>>, vector_of_vectors_to_py_list_of_lists_size_t>();

    // Expose NeuronSimulator class
    class_<NeuronSimulator>("NeuronSimulator", init<size_t, double, const std::vector<std::vector<int>> &, const std::vector<std::vector<int>> &, double, const std::vector<std::vector<double>> &, int, double, int>())
        .def("run_simulation", &NeuronSimulator::run_simulation)
        .def("get_calcium_data", &NeuronSimulator::get_calcium_data) // Returns boost::python::list
        .def("get_spike_events", &NeuronSimulator::get_spike_events); // Returns boost::python::list
}

