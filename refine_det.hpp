#include <torch/script.h> // One-stop header.
#include "torch/torch.h"
#include "torch/jit.h"
#include <iostream>
#include <memory>
#include "opencv2/opencv.hpp"
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


class RefineDet {
public:
    // load the traced module and m_prior[6375*4]
    RefineDet(const std::string path_pt);

    ~RefineDet(){}

    torch::Tensor Forward(const cv::Mat& image);

private:
    bool sub_mean(const cv::Mat &img,cv::Mat &m_out);
    bool base_transform(const cv::Mat &m_src,cv::Mat &m_out);
    bool load_img(const cv::Mat &image,torch::Tensor &input_tensor);
    torch::Tensor decode(const torch::Tensor _loc,torch::Tensor _prior);
    torch::Tensor center(torch::Tensor retv);
    bool nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor &keep, int &count,float overlap=0.5, int top_k=200);
    bool PriorBox();


private:
//    torch::jit::script::Module m_model;
    std::shared_ptr<torch::jit::script::Module> m_model;

    static const int m_SIZE_IMAGE = 320;
    static const int m_num_class = 21;//25
    static const int m_prior_size = 6375;

    torch::Tensor m_prior;//6375 * 4
};


