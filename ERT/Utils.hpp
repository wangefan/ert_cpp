

#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <nlohmann/json.hpp>

struct DataLeafInfo;
class Ferm;
class SampleData;
class Configuration;

namespace Utils {
    void startTime();

    std::chrono::microseconds getTimeStamp();
    
    std::shared_ptr<cv::Mat> translateTo(const cv::Mat& src_matrix, const cv::Mat& ts_matrix);
    std::shared_ptr<cv::Mat> computeSimilarityTransform(const cv::Mat& src_matrix, const cv::Mat& dest_matrix);
    cv::Mat_<double> json2Mat(const nlohmann::json& );
    std::vector<cv::Rect> getFaces(const cv::Mat_<uchar>& image);
    void drawLandmarks(const cv::Mat& landmark, cv::Mat_<uchar>& image, int radius, const cv::Scalar& color);
    std::vector<int> getDataLeafsNo(const std::shared_ptr<Ferm>& pFerm, const std::vector<std::shared_ptr<SampleData>>& datas);
    void resetDataLeafsIndex(const std::vector<std::shared_ptr<SampleData>>& datas);
    void adjustCurLandmarks(const std::vector<std::shared_ptr<SampleData>>& datas, 
                            const std::vector<std::shared_ptr<DataLeafInfo>>& datas_leaf_info_in_group,
                            const std::shared_ptr<Configuration>& configuration);
}
#endif