#ifndef _FERM_HPP_
#define _FERM_HPP_
#include <Configuration.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <memory>
#include <nlohmann/json.hpp>

class SampleData;

struct FermNodeInfo {
    static const std::string A_FEATURE_CLOSEST_LANDMARK_NO;
    static const std::string B_FEATURE_CLOSEST_LANDMARK_NO;
    static const std::string A_FEATURE_CLOSEST_LANDMARK_OFFSET_X;
    static const std::string A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y;
    static const std::string A_FEATURE_CLOSEST_LANDMARK_OFFSET;
    static const std::string B_FEATURE_CLOSEST_LANDMARK_OFFSET_X;
    static const std::string B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y;
    static const std::string B_FEATURE_CLOSEST_LANDMARK_OFFSET;
    static const std::string FEATURE_THRESHOLD;

    int a_feature_closest_landmark_no;
    cv::Mat_<double> a_feature_closest_landmark_offset;
    int b_feature_closest_landmark_no;
    cv::Mat_<double> b_feature_closest_landmark_offset;
    double feature_threshold;
};


class Ferm {
    static const std::string FERM_NAME;
    static const std::string FERM_NAME_VAL;
    static const std::string FERM_NODES;
    static const std::string FERM_LEAFS;
public:
    // constructor
    Ferm(int no, std::shared_ptr<Configuration> configuration);

    // member function
    int getNodesNum();
    void fromJSONOBJ(const nlohmann::json& );
    void predict(const std::vector<std::shared_ptr<SampleData>>&);
    std::shared_ptr<cv::Mat_<double>> getResidual(int leaf_no);

    // data member
    int _no;
    std::shared_ptr<Configuration> _configuration;    
    std::vector<std::shared_ptr<FermNodeInfo>> _ferm_nodes;
    std::vector<std::shared_ptr<cv::Mat_<double>>> _ferm_leafs;
    
private:
    Ferm();
    void splitNodeForPredict(int idx_ferm_node, 
                             const std::vector<std::shared_ptr<SampleData>>& datas,
                             std::shared_ptr<FermNodeInfo> pFermNodeInfo);

};
#endif
