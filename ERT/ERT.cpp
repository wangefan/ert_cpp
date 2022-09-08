#include <ERT.hpp>
#include <Utils.hpp>
#include <iostream>
#include<fstream>
#include <nlohmann/json.hpp>
#include <SampleData.hpp>

const std::string ERT::MODEL_NAME = "model_name";
const std::string ERT::LANDMARK_NUM = "landmarks_num";
const std::string ERT::CASCADE_NUM = "cascade_num";
const std::string ERT::FERM_NUM = "ferm_num";
const std::string ERT::FERM_NUM_PER_GROUP = "ferm_num_per_group";
const std::string ERT::FERM_DEPTH = "ferm_depth";
const std::string ERT::FEATURE_POOL_NUM = "feature_pool_num";
const std::string ERT::CANDIDATE_FERM_NODE_INFO_NUM = "candidate_ferm_node_infos_num";
const std::string ERT::SHRINKAGE_FAC = "shrinkage_factor";
const std::string ERT::PADDING = "padding";
const std::string ERT::LAMDA = "lamda";
const std::string ERT::MEAN_FACE = "mean_face";
const std::string ERT::REGRESSORS = "regressors";

std::unique_ptr<ERT> ERT::loadModel(const std::string& modelPath) {
	std::ifstream in(modelPath.c_str());
    nlohmann::json ert_load;
    in >> ert_load;
    auto pErt = std::make_unique<ERT>(std::make_shared<Configuration> (ert_load[ERT::LANDMARK_NUM],
                                                            1,
                                                            ert_load[ERT::CASCADE_NUM],
                                                            ert_load[ERT::FERM_NUM],
                                                            ert_load[ERT::FERM_NUM_PER_GROUP],
                                                            ert_load[ERT::FERM_DEPTH],
                                                            ert_load[ERT::CANDIDATE_FERM_NODE_INFO_NUM],
                                                            ert_load[ERT::FEATURE_POOL_NUM],
                                                            ert_load[ERT::SHRINKAGE_FAC],
                                                            ert_load[ERT::PADDING],
                                                            ert_load[ERT::LAMDA]));
    // mean face
    pErt->_mean_landmarks_normalized = Utils::json2Mat(ert_load[ERT::MEAN_FACE]);
    #ifdef _DEBUG
    std::cout << pErt->_mean_landmarks_normalized << std::endl;
    #endif
    
    // regressors
    int cascade_num = ert_load[ERT::REGRESSORS].size();
    for (int idx_regressor = 0; idx_regressor < cascade_num; ++idx_regressor) {
        auto pRegressor = pErt->_regressors[idx_regressor];
        pRegressor->fromJSONOBJ(ert_load[ERT::REGRESSORS][idx_regressor]);
    }
       
    return pErt;
}

ERT::ERT(std::shared_ptr<Configuration> configuration) {
    _configuration = configuration;
    for (int idx_regressor = 0; idx_regressor < _configuration->_cascade_number; ++ idx_regressor) {
        _regressors.push_back(std::make_shared<Regressor>(idx_regressor, _configuration));
    }
}

std::shared_ptr<cv::Mat> ERT::predict(std::shared_ptr<cv::Mat_<uchar>> pImage) {
    auto faces = Utils::getFaces(*pImage);
    if (faces.size() <= 0)
        return NULL;
    cv::Rect face = faces[0];
    auto pSampleData = std::make_shared<SampleData>(face);
    pSampleData->setNomalizedCurLandmark(_mean_landmarks_normalized);
    pSampleData->_predict_image = pImage;

    for (int idx_regressor = 0; idx_regressor < _regressors.size(); ++ idx_regressor) {
        _regressors[idx_regressor]->predict(pSampleData, _mean_landmarks_normalized);
    }
    return pSampleData->_cur_landmark;
}
