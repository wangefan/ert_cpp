#ifndef _REGRESSOR_HPP_
#define _REGRESSOR_HPP_
#include <Configuration.hpp>
#include <Ferm.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <SampleData.hpp>

struct DataLeafInfo
{
    std::shared_ptr<Ferm> pFerm;
    std::vector<int> datas_leafs_no;
};

class Regressor {
public:
    // constructor
    Regressor(int no, std::shared_ptr<Configuration> configuration);

    // member function
    void fromJSONOBJ(const nlohmann::json& );
    void predict(std::shared_ptr<SampleData> pSampleData, const cv::Mat& mean_landmarks_normalized);

    // data member
    int _no;
    std::shared_ptr<Configuration> _configuration;    
    std::vector<std::shared_ptr<Ferm>> _ferms;
private:
    Regressor();

};
#endif
