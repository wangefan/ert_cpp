#include <Regressor.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <Utils.hpp>

const std::string FERMS = "ferms";
Regressor::Regressor(int no, std::shared_ptr<Configuration> configuration) {
    #ifdef _DEBUG_
    std::cout << "Regressor " << no << " construct" << std::endl;
    #endif
    _configuration = configuration;
    for (int idx_ferm = 0; idx_ferm < _configuration->_ferm_number; ++ idx_ferm) {
        _ferms.push_back(std::make_shared<Ferm>(idx_ferm, _configuration));
    }
}

void Regressor::fromJSONOBJ(const nlohmann::json& regressorJson) {
    const nlohmann::json& ferms_json = regressorJson[FERMS];
    for (int idx_ferm_json = 0; idx_ferm_json < ferms_json.size(); ++idx_ferm_json) {
        const nlohmann::json& ferm_json = ferms_json[idx_ferm_json];
        auto pFerm = _ferms[idx_ferm_json];
        pFerm->fromJSONOBJ(ferm_json);
    }
}

void Regressor::predict(std::shared_ptr<SampleData> pSampleData, const cv::Mat& mean_landmarks_normalized) {
    // get TS from mean to cur normalized
    pSampleData->_mean_to_cur_normalized = Utils::computeSimilarityTransform(mean_landmarks_normalized, *pSampleData->_cur_landmark_normalize);

    // predict in ferms
    std::vector<std::shared_ptr<DataLeafInfo>> datas_leaf_info_in_group;
    std::vector<std::shared_ptr<SampleData>> datas;
    datas.push_back(pSampleData);

    for (int idx_ferm = 0; idx_ferm < _ferms.size(); ++idx_ferm) {
        auto pFerm = _ferms[idx_ferm];
        pFerm->predict(datas);
        std::shared_ptr<DataLeafInfo> pLeafInfo = std::make_shared<DataLeafInfo>();
        pLeafInfo->pFerm = pFerm;
        pLeafInfo->datas_leafs_no = Utils::getDataLeafsNo(pFerm, datas);
        datas_leaf_info_in_group.push_back(pLeafInfo);
        Utils::resetDataLeafsIndex(datas);
        if(datas_leaf_info_in_group.size() >= _configuration->_ferm_num_per_group) {
            Utils::adjustCurLandmarks(datas, datas_leaf_info_in_group, _configuration);
            datas_leaf_info_in_group.clear();
        }
    }
}
