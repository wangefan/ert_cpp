#include <memory>
#include <ERT.hpp>
#include <Utils.hpp>


 
int main() {
    // 1. load model
    const std::string modelPath = "./ert_model_good.json";
    std::unique_ptr<ERT> pErt = ERT::loadModel(modelPath);

    // 2. load video
    const std::string test_video_path = "../face.mp4";
    cv::VideoCapture cap(test_video_path);
  	if(!cap.isOpened()) {
    	std::cout << "Vedio open failed. please check your vedio equitment." << std::endl;
    	exit(0);
  	}
  	std::cout << "open vedio." << std::endl;

    cv::Mat_<uchar> image;
    std::shared_ptr<cv::Mat_<uchar>> pGray = std::make_shared<cv::Mat_<uchar>>();
    while (true) {
        cap >> image;
        Utils::startTime();
        cv::cvtColor(image, *pGray, cv::COLOR_BGR2GRAY);
        auto pLandmark = pErt->predict(pGray);
        if(pLandmark != NULL)
          Utils::drawLandmarks(*pLandmark, image, 2, cv::Scalar(0, 255, 255));
        auto diff = Utils::getTimeStamp();
        std::cout<<"diff = " << diff.count() << " us" << std::endl;
        cv::imshow("face", image);   
        cv::waitKey(1);
    }
    
    
    return 0;
}
