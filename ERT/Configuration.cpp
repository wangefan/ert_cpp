#include <Configuration.hpp>

Configuration::Configuration(int num_landmarks,
               int train_data_times,
               int cascade_number,
               int ferm_number,
               int ferm_group_number,
               int ferm_depth,
               int num_candidate_ferm_node_infos,
               int feature_pool_size,
               float shrinkage_factor,
               float padding,
               float lamda) {
    _num_landmarks = num_landmarks;
    _train_data_times = train_data_times;
    _cascade_number = cascade_number;
    _ferm_number = ferm_number;
    _ferm_num_per_group = ferm_group_number;
    _ferm_depth = ferm_depth;
    _num_candidate_ferm_node_infos = num_candidate_ferm_node_infos;
    _feature_pool_size = feature_pool_size;
    _shrinkage_factor = shrinkage_factor;
    _padding = padding;
    _lamda = lamda;
}