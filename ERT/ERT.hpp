#ifndef _ERT_HPP_
#define _ERT_HPP_

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Configuration.hpp>
#include <Regressor.hpp>

class ERT {
public:
    static const std::string MODEL_NAME;
    static const std::string LANDMARK_NUM;
    static const std::string CASCADE_NUM;
    static const std::string FERM_NUM;
    static const std::string FERM_NUM_PER_GROUP;
    static const std::string FERM_DEPTH;
    static const std::string FEATURE_POOL_NUM;
    static const std::string CANDIDATE_FERM_NODE_INFO_NUM;
    static const std::string SHRINKAGE_FAC;
    static const std::string PADDING;
    static const std::string LAMDA;
    static const std::string MEAN_FACE;
    static const std::string REGRESSORS;
    
    // static function
    static std::unique_ptr<ERT> loadModel(const std::string& modelPath);

    // constructor
    ERT(std::shared_ptr<Configuration>);

    // data member
    std::shared_ptr<Configuration> _configuration;
    cv::Mat_<double> _mean_landmarks_normalized;
    std::vector<std::shared_ptr<Regressor>> _regressors;

    // member function
    std::shared_ptr<cv::Mat> predict(std::shared_ptr<cv::Mat_<uchar>> pImage);
};

#endif