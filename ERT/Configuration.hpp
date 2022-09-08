#ifndef _CONFIGURATION_HPP_
#define _CONFIGURATION_HPP_

#include <string>

class Configuration {
private:
    Configuration();
public:
    Configuration(int num_landmarks,
               int train_data_times,
               int cascade_number,
               int ferm_number,
               int ferm_group_number,
               int ferm_depth,
               int num_candidate_ferm_node_infos,
               int feature_pool_size,
               float shrinkage_factor,
               float padding,
               float lamda);
    
    int _num_landmarks;
    int _train_data_times;
    int _cascade_number;
    int _ferm_number;
    int _ferm_num_per_group;
    int _ferm_depth;
    int _num_candidate_ferm_node_infos;
    int _feature_pool_size;
    float _shrinkage_factor;
    float _padding;
    float _lamda;

};

#endif