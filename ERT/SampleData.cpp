#include <SampleData.hpp>
#include <iostream>
#include <Utils.hpp>

const static cv::Mat NOR_MAT = (cv::Mat_<double>(4,2) << 0, 0, 
                                                           1, 0, 
                                                           0, 1, 
                                                           1, 1);

SampleData::SampleData(const cv::Rect& face): _face(face),
                                              _ferm_node_index(0) {
    #ifdef _DEBUG_
    std::cout << "SampleData " << " construct" << std::endl;
    #endif

    cv::Mat_<double> faceBox(4, 2);
    faceBox(0, 0) = _face.x; faceBox(0, 1) = _face.y; 
    faceBox(1, 0) = _face.x + _face.width; faceBox(1, 1) = _face.y; 
    faceBox(2, 0) = _face.x; faceBox(2, 1) = _face.y + _face.height; 
    faceBox(3, 0) = _face.x+ _face.width; faceBox(3, 1) = _face.y + _face.height; 

    _nor_matrix = Utils::computeSimilarityTransform(faceBox, NOR_MAT);
    _unnor_matrix = Utils::computeSimilarityTransform(NOR_MAT, faceBox);
}

void SampleData::setNomalizedCurLandmark(const cv::Mat& src_normalized_landmark) {
    _cur_landmark_normalize = std::make_shared<cv::Mat>(src_normalized_landmark.clone());
    _cur_landmark = Utils::translateTo(*_cur_landmark_normalize, *_unnor_matrix);
}


