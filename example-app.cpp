#include"refine_det.hpp"

vector<string> label_map = {
          "bg",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};


int main(int argc, const char* argv[])
{
    string path_pt = "/data_1/refinedet320_20200520_pytorch1_0.pt";

    RefineDet refdet(path_pt);
    fstream infile("/data_1/pic.txt");
    string img_path;
    int cnt=0;
    while(infile >> img_path)
    {
        std::cout<<cnt++<<"   img_path="<<img_path<<std::endl;

        cv::Mat img = cv::imread(img_path);

        auto t0 = std::chrono::steady_clock::now();
        torch::Tensor result = refdet.Forward(img);
        std::cout << "consume time="<<std::chrono::duration_cast<std::chrono::milliseconds>
                     (std::chrono::steady_clock::now() - t0).count()<<"ms"<<std::endl;

        // x1,y1,x2,y2,score,id
        auto result_data = result.accessor<double, 2>();
        cv::Mat img_draw = img.clone();
        for(int i=0;i<result_data.size(0);i++)
        {
            float score = result_data[i][4];
            if(score < 0.4) { continue;}
            int x1 = result_data[i][0];
            int y1 = result_data[i][1];
            int x2 = result_data[i][2];
            int y2 = result_data[i][3];
            int id_label = result_data[i][5];

            cv::rectangle(img_draw,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,0),3);
            cv::putText(img_draw,label_map[id_label],cv::Point(x1,y2),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,55));
        }

        cv::namedWindow("img_draw",0);
        cv::imshow("img_draw",img_draw);
        cv::waitKey(0);
    }

    return 0;
}
