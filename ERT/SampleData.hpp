#ifndef _SAMPLEDATA_HPP_
#define _SAMPLEDATA_HPP_


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <memory>


class SampleData {
public:
    // constructor
    SampleData(const cv::Rect& face);

    // member function
    void setNomalizedCurLandmark(const cv::Mat& src_normalized_landmark);

    // data member
    cv::Rect _face;
    std::shared_ptr<cv::Mat_<uchar>> _predict_image;
    std::shared_ptr<cv::Mat> _nor_matrix;
    std::shared_ptr<cv::Mat> _unnor_matrix;
    std::shared_ptr<cv::Mat> _mean_to_cur_normalized;
    std::shared_ptr<cv::Mat> _cur_landmark_normalize;
    std::shared_ptr<cv::Mat> _cur_landmark;
    int _ferm_node_index;
    
private:
    SampleData();

};
#endif
