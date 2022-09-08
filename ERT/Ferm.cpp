#include <Ferm.hpp>
#include <iostream>
#include <Utils.hpp>
#include <SampleData.hpp>

const std::string FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_NO = "a_feature_closest_landmark_no";
const std::string FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_OFFSET_X = "a_feature_closest_landmark_offset_x";
const std::string FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y = "a_feature_closest_landmark_offset_y";
const std::string FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_OFFSET = "a_feature_closest_landmark_offset";
const std::string FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_NO = "b_feature_closest_landmark_no";
const std::string FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_OFFSET_X = "b_feature_closest_landmark_offset_x";
const std::string FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y = "b_feature_closest_landmark_offset_y";
const std::string FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_OFFSET = "b_feature_closest_landmark_offset";
const std::string FermNodeInfo::FEATURE_THRESHOLD = "feature_threshold";

const std::string Ferm::FERM_NAME = "ferm_name";
const std::string Ferm::FERM_NAME_VAL = "ferm";
const std::string Ferm::FERM_NODES = "ferm_nodes";
const std::string Ferm::FERM_LEAFS = "ferm_leafs";

Ferm::Ferm(int no, std::shared_ptr<Configuration> configuration) {
    #ifdef _DEBUG_
    std::cout << "Ferm " << no << " construct" << std::endl;
    #endif
    _no = no;
    _configuration = configuration;

    int num_ferm_node = std::pow(2, _configuration->_ferm_depth - 1) - 1;
    for (int idx_ferm_node = 0; idx_ferm_node < num_ferm_node; ++ idx_ferm_node) {
        _ferm_nodes.push_back(std::make_shared<FermNodeInfo>());
    }
    int num_ferm_leaf = std::pow(2, _configuration->_ferm_depth - 1);
    /*for (int idx_ferm_leaf = 0; idx_ferm_leaf < num_ferm_leaf; ++ idx_ferm_leaf) {
        _ferm_leafs.push_back(std::make_shared<cv::Mat_<double>>());
    }*/
}

int Ferm::getNodesNum() {
    return _ferm_nodes.size();
}

// member function
void Ferm::fromJSONOBJ(const nlohmann::json& fermJson) {
    int num_node_infos = fermJson[FERM_NODES].size();
    for (int idx_ferm_node = 0; idx_ferm_node < num_node_infos; ++ idx_ferm_node) {
        auto pFermNode = _ferm_nodes[idx_ferm_node];
        const nlohmann::json& fermNodeJson = fermJson[FERM_NODES].at(idx_ferm_node);
        pFermNode->a_feature_closest_landmark_no = fermNodeJson[FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_NO];
        pFermNode->a_feature_closest_landmark_offset = (cv::Mat_<double>(1, 2) << 
                                        fermNodeJson[FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                        fermNodeJson[FermNodeInfo::A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y]);
        pFermNode->b_feature_closest_landmark_no = fermNodeJson[FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_NO];
        pFermNode->b_feature_closest_landmark_offset = (cv::Mat_<double>(1, 2) << 
                                        fermNodeJson[FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                        fermNodeJson[FermNodeInfo::B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y]);                                
        pFermNode->feature_threshold = fermNodeJson[FermNodeInfo::FEATURE_THRESHOLD];
    }

    const nlohmann::json& fermLeafsJson = fermJson[FERM_LEAFS];
    int num_ferm_leafs = fermLeafsJson.size();
    for (int idx_ferm_leaf = 0; idx_ferm_leaf < num_ferm_leafs; ++ idx_ferm_leaf) {
        const nlohmann::json& fermLeafJson = fermLeafsJson[idx_ferm_leaf];
        auto pFermLeaf = std::make_shared<cv::Mat_<double>>();
        *pFermLeaf = Utils::json2Mat(fermLeafJson);
        _ferm_leafs.push_back(pFermLeaf);
        
    }
}

void Ferm::splitNodeForPredict(int idx_ferm_node, 
                               const std::vector<std::shared_ptr<SampleData>>& datas,
                               std::shared_ptr<FermNodeInfo> pFermNodeInfo) {
    for(std::shared_ptr<SampleData> pData : datas) {
        if(idx_ferm_node == pData->_ferm_node_index) {
            cv::Mat rotate_scale_mat = (*(pData->_mean_to_cur_normalized))(cv::Rect(0, 0, 2, 2));

            // get A/B offset from mean to cur normalized.
            cv::Mat A_feature_closest_landmark_offset_cur_nor = 
                pFermNodeInfo->a_feature_closest_landmark_offset * rotate_scale_mat;
            cv::Mat B_feature_closest_landmark_offset_cur_nor = 
                pFermNodeInfo->b_feature_closest_landmark_offset * rotate_scale_mat;

            // get A/B cur landmark normalized position 
            cv::Mat A_cur_landmark_nor = (pData->_cur_landmark_normalize->rowRange(pFermNodeInfo->a_feature_closest_landmark_no, pFermNodeInfo->a_feature_closest_landmark_no+1));
            cv::Mat A_cur_feature_nor = 
                A_feature_closest_landmark_offset_cur_nor + A_cur_landmark_nor;
            cv::Mat B_cur_landmark_nor = (pData->_cur_landmark_normalize->rowRange(pFermNodeInfo->b_feature_closest_landmark_no, pFermNodeInfo->b_feature_closest_landmark_no+1));
            cv::Mat B_cur_feature_nor = 
                B_feature_closest_landmark_offset_cur_nor + B_cur_landmark_nor;
            
            // get A/B cur feature pos
            auto A_feature_cur = Utils::translateTo(A_cur_feature_nor, *pData->_unnor_matrix);
            auto B_feature_cur = Utils::translateTo(B_cur_feature_nor, *pData->_unnor_matrix);
            
            int A_feature_pixel = 0, B_feature_pixel = 0;
            int A_feature_pixel_pos_x = (int)(*A_feature_cur).at<double>(0, 0);
            int A_feature_pixel_pos_y = (int)(*A_feature_cur).at<double>(0, 1);
            int B_feature_pixel_pos_x = (int)(*B_feature_cur).at<double>(0, 0);
            int B_feature_pixel_pos_y = (int)(*B_feature_cur).at<double>(0, 1);
            
            // get A/B pixel value
            if(A_feature_pixel_pos_x >= 0 && A_feature_pixel_pos_x < pData->_predict_image->cols &&
               A_feature_pixel_pos_y >= 0 && A_feature_pixel_pos_y < pData->_predict_image->rows ) {
                   A_feature_pixel = pData->_predict_image->at<uchar>(A_feature_pixel_pos_y, A_feature_pixel_pos_x);
            }
            if(B_feature_pixel_pos_x >= 0 && B_feature_pixel_pos_x < pData->_predict_image->cols &&
               B_feature_pixel_pos_y >= 0 && B_feature_pixel_pos_y < pData->_predict_image->rows ) {
                   B_feature_pixel = pData->_predict_image->at<uchar>(B_feature_pixel_pos_y, B_feature_pixel_pos_x);
            }

            // splite 
            int left_node_index = 2 * idx_ferm_node + 1;
            int right_node_index = 2 * idx_ferm_node + 2;
            if(double(A_feature_pixel - B_feature_pixel) > pFermNodeInfo->feature_threshold) {
                pData->_ferm_node_index = left_node_index;
            } else {
                pData->_ferm_node_index = right_node_index;
            }
        }
    }
}

void Ferm::predict(const std::vector<std::shared_ptr<SampleData>>& datas) {
    for (int idx_ferm_node = 0; idx_ferm_node < _ferm_nodes.size(); ++ idx_ferm_node) {
        splitNodeForPredict(idx_ferm_node,
                           datas,
                           _ferm_nodes[idx_ferm_node]);
    }
}

std::shared_ptr<cv::Mat_<double>> Ferm::getResidual(int leaf_no) {
    if(_ferm_leafs.size() > 0 && _ferm_leafs.size() > leaf_no) {
        return _ferm_leafs[leaf_no];
    }
    return NULL;
}