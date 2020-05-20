#include "refine_det.hpp"

RefineDet::RefineDet(const std::string path_pt)
{
    try{
        torch::Device m_device(torch::kCUDA);
        m_model = torch::jit::load(path_pt);
        m_model->to(m_device);
    }
    catch (...)
    {
        std::cout<<"********************************catch-error***********************************"<<std::endl;
        std::cout<<"load model error!"<<std::endl;
        std::cout<<"please check model path or cuda:"<<std::endl;
        std::cout<<"model path="<<path_pt<<std::endl;
        std::cout<<"********************************catch-error***********************************\n"<<std::endl;
    }

    PriorBox();
    //    m_model->eval();
}

bool RefineDet::sub_mean(const cv::Mat &img,cv::Mat &m_out)
{
    const vector<float> m_v_mean = {104.0,117.0,123.0};
    if(3 != img.channels() || 3 != m_v_mean.size() || img.empty())
    {
        return false;
    }
    cv::Mat m_arr[3];
    cv::split(img,m_arr);
    for(int i=0;i<m_v_mean.size();i++)
    {
        m_arr[i] = m_arr[i] - m_v_mean[i];
    }
    merge(m_arr,3,m_out);
    return true;
}

bool RefineDet::base_transform(const cv::Mat &m_src,cv::Mat &m_out)
{
    cv::Mat image;
    cv::resize(m_src,image,cv::Size(m_SIZE_IMAGE,m_SIZE_IMAGE));
    image.convertTo(image, CV_32FC3);
    bool b_flg = sub_mean(image,m_out);
    if(!b_flg)
    {
        return false;
    }
    cv::cvtColor(m_out,m_out,CV_BGR2RGB);
    m_out.convertTo(m_out, CV_32FC3);
    return true;
}

bool RefineDet::load_img(const cv::Mat &image,torch::Tensor &input_tensor)
{
    torch::Device m_device(torch::kCUDA);
    if(image.empty()) { return false;}
    cv::Mat m_out;
    base_transform(image,m_out);

    //[320,320,3]
    input_tensor = torch::from_blob(
                m_out.data, {m_SIZE_IMAGE, m_SIZE_IMAGE, 3}).toType(torch::kFloat32);//torch::kByte //大坑
    //[3,320,320]
    input_tensor = input_tensor.permute({2,0,1});
    input_tensor = input_tensor.unsqueeze(0);
    input_tensor = input_tensor.to(torch::kFloat).to(m_device);
    return true;
}


bool RefineDet::PriorBox()
{
    std::vector<float> mean;
    std::vector<int> feature_maps = {40,20,10,5};
    int image_size = 320;
    vector<int> steps = {8,16,32,64};
    vector<int> min_sizes = {32,64,128,256};
    vector<int> aspect_ratios = {2,2,2,2};
    for(int k=0;k<feature_maps.size();k++)
    {
        int f = feature_maps[k];
        for(int i=0;i<f;i++)
        {
            for(int j=0;j<f;j++)
            {
                float f_k = image_size * 1.0 / steps[k];
                float cx = (j + 0.5) / f_k;
                float cy = (i + 0.5) / f_k;
                float s_k = min_sizes[k] * 1.0 / image_size;
                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k);
                mean.push_back(s_k);

                float ar = aspect_ratios[k];
                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k * 1.0 * sqrt(ar));
                mean.push_back(s_k * 1.0 / sqrt(ar));

                mean.push_back(cx);
                mean.push_back(cy);
                mean.push_back(s_k * 1.0 / sqrt(ar));
                mean.push_back(s_k * 1.0 * sqrt(ar));
            }
        }
    }
    m_prior = torch::from_blob(mean.data(),{m_prior_size,4}).cuda();
    m_prior = m_prior.clamp(0,1);
    //    std::cout<<m_prior<<std::endl;
    return true;
}

torch::Tensor RefineDet::decode(const torch::Tensor _loc,torch::Tensor _prior)
{
    vector<float> variance({0.1,0.2});
    auto top_2 = torch::tensor({0,1}).cuda().to(torch::kLong);
    auto bottom_2 = torch::tensor({2,3}).cuda().to(torch::kLong);

    auto c1 = _prior.index_select(1,top_2)+_loc.index_select(1,top_2).mul(variance[0])*_prior.index_select(1,bottom_2);
    auto c2 = _prior.index_select(1,bottom_2)*torch::exp(_loc.index_select(1,bottom_2)*variance[1]);
    auto _retv = torch::cat({c1,c2},1);
    auto c3 = _retv.index_select(1,top_2)-_retv.index_select(1,bottom_2).div(2);
    auto c4 = c3 + _retv.index_select(1,bottom_2);
    return torch::cat({c3,c4},1);
}

torch::Tensor RefineDet::center(torch::Tensor retv)
{
    auto c1 = retv.select(1,0).unsqueeze(1);
    auto c2 = retv.select(1,1).unsqueeze(1);
    auto c3 = retv.select(1,2).unsqueeze(1);
    auto c4 = retv.select(1,3).unsqueeze(1);

    auto _retv = torch::cat({(c1+c3).div(2),(c2+c4).div(2),c3-c1,c4-c2},1);
    return _retv;
}

bool RefineDet::nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor &keep, int &count,float overlap, int top_k)
{
    count =0;
    keep = torch::zeros({scores.size(0)}).to(torch::kLong).to(scores.device());
    if(0 == boxes.numel())
    {
        return false;
    }

    torch::Tensor x1 = boxes.select(1,0).clone();
    torch::Tensor y1 = boxes.select(1,1).clone();
    torch::Tensor x2 = boxes.select(1,2).clone();
    torch::Tensor y2 = boxes.select(1,3).clone();
    torch::Tensor area = (x2-x1)*(y2-y1);
    //    std::cout<<area<<std::endl;

    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
    torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

    int num_ = idx.size(0);
    if(num_ > top_k) //python:idx = idx[-top_k:]
    {
        idx = idx.slice(0,num_-top_k,num_).clone();
    }
    torch::Tensor xx1,yy1,xx2,yy2,w,h;
    while(idx.numel() > 0)
    {
        auto i = idx[-1];
        keep[count] = i;
        count += 1;
        if(1 == idx.size(0))
        {
            break;
        }
        idx = idx.slice(0,0,idx.size(0)-1).clone();

        xx1 = x1.index_select(0,idx);
        yy1 = y1.index_select(0,idx);
        xx2 = x2.index_select(0,idx);
        yy2 = y2.index_select(0,idx);

        xx1 = xx1.clamp(x1[i].item().toFloat(),INT_MAX*1.0);
        yy1 = yy1.clamp(y1[i].item().toFloat(),INT_MAX*1.0);
        xx2 = xx2.clamp(INT_MIN*1.0,x2[i].item().toFloat());
        yy2 = yy2.clamp(INT_MIN*1.0,y2[i].item().toFloat());

        w = xx2 - xx1;
        h = yy2 - yy1;

        w = w.clamp(0,INT_MAX);
        h = h.clamp(0,INT_MAX);

        torch::Tensor inter = w * h;
        torch::Tensor rem_areas = area.index_select(0,idx);

        torch::Tensor union_ = (rem_areas - inter) + area[i];
        torch::Tensor Iou = inter * 1.0 / union_;
        torch::Tensor index_small = Iou < overlap;
        auto mask_idx = torch::nonzero(index_small).squeeze();
        idx = idx.index_select(0,mask_idx);//pthon: idx = idx[IoU.le(overlap)]
    }
    return true;
}


torch::Tensor RefineDet::Forward(const cv::Mat& image)
{
    torch::Tensor input_tensor;
    load_img(image,input_tensor);

    auto output = m_model->forward({input_tensor});

    auto tpl = output.toTuple();
    auto arm_loc = tpl->elements()[0].toTensor();
    // arm_loc.print();
    //    std::cout<<arm_loc[0]<<std::endl;
    auto arm_conf = tpl->elements()[1].toTensor();
    //arm_conf.print();
    auto odm_loc = tpl->elements()[2].toTensor();
    //odm_loc.print();
    //     std::cout<<odm_loc[0]<<std::endl;
    auto odm_conf = tpl->elements()[3].toTensor();
    //    odm_conf.print();


    float obj_threshed = 0.01;
    torch::Tensor arm_object_conf = arm_conf.squeeze(0).select(1,1);
    torch::Tensor object_index = arm_object_conf > obj_threshed;
    object_index=object_index.unsqueeze(1);

    torch::Tensor object_index_1 = object_index.expand_as(odm_conf.squeeze(0)).toType(torch::kFloat64);
    auto filter_odm_conf = odm_conf.squeeze(0).toType(torch::kFloat64) * object_index_1;
    torch::Tensor conf_preds_ = filter_odm_conf.clone().toType(torch::kFloat64);
    torch::Tensor conf_preds = conf_preds_.transpose(1,0).toType(torch::kFloat64);
    torch::Tensor default_m = decode(arm_loc[0],m_prior);
    default_m = center(default_m);
    torch::Tensor decode_boxes_m = decode(odm_loc[0],default_m);//6375,4

    float conf_thresh = 0.01;

    cv::Mat img_draw = image.clone();
    int height = img_draw.rows;
    int width = img_draw.cols;

    torch::Tensor result_out;
    for(int i=1;i<m_num_class;i++)
    {
        torch::Tensor c_mask_m = conf_preds[i] > 0.01;
        torch::Tensor nonzero_index = torch::nonzero(c_mask_m);
        torch::Tensor  score_m = torch::index_select(conf_preds[i],0,nonzero_index.squeeze(1));
        torch::Tensor  boxes_m = torch::index_select(decode_boxes_m,0,nonzero_index.squeeze(1));

        torch::Tensor keep;
        int count = 0;
        float overlap = 0.45;
        int top_k=1000;
        nms(boxes_m, score_m, keep, count, overlap, top_k);
        if(0 == count) { continue; }

        keep = keep.slice(0,0,count).clone();
        torch::Tensor score_my = score_m.index_select(0,keep);
        torch::Tensor boxes_my = boxes_m.index_select(0,keep);

        if(score_my[0].item().toFloat() < conf_thresh)
        {
            continue;
        }
        boxes_my.select(1,0).mul_(width);
        boxes_my.select(1,1).mul_(height);
        boxes_my.select(1,2).mul_(width);
        boxes_my.select(1,3).mul_(height);
        torch::Tensor label_tensor = torch::full_like(score_my.unsqueeze(1),i);
        torch::Tensor result_ = torch::cat({boxes_my.toType(torch::kFloat64),score_my.unsqueeze(1).toType(torch::kFloat64),label_tensor.toType(torch::kFloat64)},1);
        if(0 == result_out.numel())
        {
            result_out = result_.clone();
        }else
        {
            result_out = torch::cat({result_out,result_},0);//按行拼接
        }
    }

    result_out =result_out.cpu();
    return result_out;
}
