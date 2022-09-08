#include <Utils.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <SampleData.hpp>
#include <Configuration.hpp>
#include <Regressor.hpp>

namespace Utils {
std::chrono::time_point<std::chrono::steady_clock> g_start_time;

void startTime() {
    g_start_time = std::chrono::steady_clock::now();
}

std::chrono::microseconds getTimeStamp() {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - g_start_time);
}

std::shared_ptr<cv::Mat> translateTo(const cv::Mat& src_matrix, const cv::Mat& ts_matrix) {
    int rows = src_matrix.rows;
    cv::Mat_<double> src_matrix_append_one_col;
    cv::hconcat(src_matrix, cv::Mat_<double>::ones(rows, 1), src_matrix_append_one_col);
    
    auto result = std::make_shared<cv::Mat>(src_matrix_append_one_col * ts_matrix);
    return result;
}

std::shared_ptr<cv::Mat> computeSimilarityTransform(const cv::Mat& src_matrix, const cv::Mat& dest_matrix) {
    int rows = src_matrix.rows;
    cv::Mat_<double> src_matrix_append_one_col;
    cv::hconcat(src_matrix, cv::Mat_<double>::ones(rows, 1), src_matrix_append_one_col);

    cv::Mat_<double> pinv;
    cv::invert(src_matrix_append_one_col, pinv, cv::DECOMP_SVD);

    auto result = std::make_shared<cv::Mat>(pinv * dest_matrix);
    return result;
}

cv::Mat_<double> json2Mat(const nlohmann::json& jsonObj) {
    const std::vector<std::vector<double>> vect = jsonObj.get<std::vector<std::vector<double>>>();
    cv::Mat mtx = cv::Mat(vect.size(), vect[0].size(), cv::DataType<double>::type);  // don't need to init??

    // copy data
    for (int i = 0; i < vect.size(); i++) 
        for (int j=0; j<vect[i].size(); j++) {
            mtx.at<double>(i,j) = vect[i][j];
        }   

    return mtx;
}

/**************************************************************
* return the numpy array face rect[l, t, w, h] like:
* [ 80  98 151 151]
**************************************************************/
std::vector<cv::Rect> getFaces(const cv::Mat_<uchar>& image) {
    std::vector<cv::Rect> faces;
    const std::string haar_feature = "./CascadeClassifier/haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load(haar_feature);
	haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
    return faces;
}

void drawLandmarks(const cv::Mat& landmark, cv::Mat_<uchar>& image, int radius, const cv::Scalar& color) {
    int landmark_number = landmark.rows;
    for(int i = 0; i < landmark_number; ++i) {
        double x = landmark.at<double>(i, 0);
        double y = landmark.at<double>(i, 1);
        cv::circle(image, cv::Point(x, y), radius, color, -1);
    }
}

std::vector<int> getDataLeafsNo(const std::shared_ptr<Ferm>& pFerm, const std::vector<std::shared_ptr<SampleData>>& datas) {
    std::vector<int> result;
    for(std::shared_ptr<SampleData> pData : datas) {
        result.push_back(pData->_ferm_node_index - pFerm->getNodesNum());
    }
    return result;
}

void resetDataLeafsIndex(const std::vector<std::shared_ptr<SampleData>>& datas) {
    for(std::shared_ptr<SampleData> pData : datas) {
        pData->_ferm_node_index = 0;
    }
}

void adjustCurLandmarks(const std::vector<std::shared_ptr<SampleData>>& datas, 
                            const std::vector<std::shared_ptr<DataLeafInfo>>& datas_leaf_info_in_group,
                            const std::shared_ptr<Configuration>& configuration) {
    for(int idx_data = 0; idx_data < datas.size(); ++ idx_data) {
        auto pData = datas[idx_data];
        cv::Mat_<double> data_residual = cv::Mat_<double>::zeros(configuration->_num_landmarks, 2);
        for (std::shared_ptr<DataLeafInfo> pLeafInfo : datas_leaf_info_in_group) {
            auto pFerm = pLeafInfo->pFerm;
            int data_leaf_no = pLeafInfo->datas_leafs_no[idx_data];
            auto itr_residual = pFerm->getResidual(data_leaf_no);
            data_residual += (*itr_residual);
        }
        data_residual /= datas_leaf_info_in_group.size();
        cv::Mat_<double> temp_cur_landmark_normalize = pData->_cur_landmark_normalize->clone();
        temp_cur_landmark_normalize += configuration->_shrinkage_factor * data_residual;
        pData->setNomalizedCurLandmark(temp_cur_landmark_normalize);
    }
    
}

}