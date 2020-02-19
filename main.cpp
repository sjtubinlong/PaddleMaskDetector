
#include <string>
#include <vector>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <paddle_inference_api.h>

bool g_enable_gpu = false;
// 检测过滤阈值
float g_threshold = 0.2;

// 用于记录检测结果
class DetectionOut {
public:
    // 图片序号
    int image_id;
    // 方框列表
    std::vector<std::vector<int>> rectange;
    // 通过opencv截取框到cv::Mat
    std::vector<cv::Mat> mats;
    // 构造函数
    DetectionOut(int idx) : image_id(idx) {        
    }
    // 获取检测库数量
    int get_rect_num() const {
        return rectange.size();
    }
    // 获取cv::Mat列表
    std::vector<cv::Mat> get_mats() {
        return mats;
    }
    // 获取检测库列表
    const std::vector<std::vector<int>>& get_rects() const {
        return rectange;
    }
    // 添加一个检测框结果
    void add_rect(std::vector<int>& rect, cv::Mat im) {
        rectange.emplace_back(rect);
        mats.push_back(im);
    }
};

// 图片预处理: 输入直接为cv::Mat
bool preprocess_image(cv::Mat im, float* buffer, const std::vector<int>& input_shape) {
    if (im.data == nullptr || im.empty()) {
        printf("Invalid Mat input\n");
        return false;
    }
    // resize
    int rc = im.channels();
	int rw = im.cols;
    int rh = im.rows;
    cv::Size resize_size(input_shape[3], input_shape[2]);
    if (rw != input_shape[3] || rh != input_shape[2]) {
        cv::resize(im, im, resize_size);
    }
    rc = im.channels();
	rw = im.cols;
    rh = im.rows;
    // 减均值除方差: (img - mean) * scale
    float mean[3] = {104, 117, 123};
    float scale[3] = {0.007843, 0.007843, 0.007843};
    //#pragma omp parallel for
    for (int h = 0; h < rh; ++h) {
        auto uptr = im.ptr<uchar>(h);
        int im_index = 0;
        for (int w = 0; w < rw; ++w) {
            for (int c = 0; c < rc; ++c) {
                int top_index = (c * rh + h) * rw + w;
                float pixel = static_cast<float>(uptr[im_index++]);
                pixel = (pixel - mean[c]) * scale[c];
                buffer[top_index] = pixel;
            }
        }
    }
    return true;
}

// 图片预处理：输入为图片路径
bool preprocess_image(std::string filename, float* buffer, const std::vector<int>& input_shape) {
        cv::Mat im = cv::imread(filename);
        if (im.data == nullptr || im.empty()) {
            printf("Fail to open image file : [%s]\n", filename.c_str());
            return false;
        }
        im.convertTo(im, CV_32FC3, 1 / 255.0);
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        return preprocess_image(im, buffer, input_shape);
}

// 用于分类的图片批量预处理
template<typename T>
bool preprocess_batch_classify(
        const std::vector<T>& images,
        std::vector<float>& input_data,
        std::vector<int>& input_shape) {
    // batch 大小
    int batch_size = input_shape[0];
    int item_size = input_shape[1] * input_shape[2] * input_shape[3];
    input_data.resize(batch_size * item_size);
    // 多线程并行预处理处理数据
    std::vector<std::thread> threads;
    for (int i = 0; i < images.size(); ++i) {
        T im = images[i];
        float* base = input_data.data();
        float* buffer = reinterpret_cast<float*>(base + i * item_size);
        threads.emplace_back([im, buffer, input_shape] {
            preprocess_image(im, buffer, input_shape);

        });
    }
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    return true;
}

// 获取所有图片大小和cv::Mat，输入为图片路径列表
void get_image_data(
        std::vector<std::string>& images,
        std::vector<cv::Mat>& mats,
        std::vector<std::vector<int>>& input_shapes)
{
    for (int i = 0; i < images.size(); ++i) {
        //cv::IMREAD_COLOR
        auto im = cv::imread(images[i], cv::IMREAD_COLOR);
        if (im.data == nullptr || im.empty()) {
            printf("Fail to open image[%d]: [%s]\n", i, images[i].c_str());
            continue;
        }
        cv::cvtColor(im, im, cv::COLOR_RGB2BGR);
        input_shapes[i] = {1, im.channels(), im.rows, im.cols};
        mats[i] = im.clone();
    }
}

// 用于检测的图片批量预处理
bool preprocess_batch_detection(
        std::vector<std::string>& images,
        std::vector<float>& input_data,
        std::vector<int>& input_shape,
        std::vector<cv::Mat>& input_mat) {
    // batch 大小
    int batch_size = images.size();
    // 用于临时每张图片预处理后的数据
    std::vector<std::vector<float>> data(batch_size);
    // 所有图片的尺寸
    std::vector<std::vector<int>> shapes(batch_size);
    // 所有图片读入到cv::Mat
    input_mat.resize(batch_size);
    get_image_data(images, input_mat, shapes);
    for (int i = 0; i < batch_size; ++i) {
        int item_size = shapes[i][1] * shapes[i][2] * shapes[i][3];
        data[i].resize(item_size);
    }
    // 多线程并行预处理处理数据
    std::vector<std::thread> threads;
    for (int i = 0; i < batch_size; ++i) {
        auto im = input_mat[i];
        auto buffer = data[i].data();
        auto shape = shapes[i];
        threads.emplace_back([im, buffer, &shape] {
            preprocess_image(im, buffer, shape);
        });
    }
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    // 找到所有图片的最大宽到高, 然后一一进行 padding
    int max_h = -1;
    int max_w = -1;
    for (int i = 0; i < batch_size; ++i) {
        max_h = (max_h > shapes[i][2] ? max_h : shapes[i][2]);
        max_w = (max_w > shapes[i][3] ? max_w : shapes[i][3]);
    }
    // 开始 padding
    input_data.clear();
    input_shape = {batch_size, 3, max_h, max_w};
    int input_max_size = batch_size * 3 * max_h * max_w;
    input_data.insert(input_data.end(), input_max_size, 0);
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        float *data_i = input_data.data() + i * 3 * max_h * max_w;
        float *lod_ptr = data[i].data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < shapes[i][2]; ++h) {
                memcpy(data_i, lod_ptr, shapes[i][3] * sizeof(float));
                lod_ptr += shapes[i][3];
                data_i += max_w;
            }
            data_i += (max_h - shapes[i][2]) * max_w;
        }
    }
    return true;
}

// 对图片进行预测
 void run_predict(std::string model_dir,
            const std::vector<float>& input_data,
            const std::vector<int>& input_shape,
            std::vector<float>& output_data,
            int output_id = 0,
            std::vector<std::vector<size_t>>* lod_data = nullptr) {
    // 设置模型配置
    paddle::AnalysisConfig config;
    config.SetModel(model_dir + "/__model__",
                    model_dir + "/__params__");
	if (g_enable_gpu) {
	    config.EnableUseGpu(100, 0);
	} else {
	    config.DisableGpu();
	    config.EnableMKLDNN();
	    config.SetCpuMathLibraryNumThreads(10);
    }
	// 使用 ZeroCopyTensor 必须设置 config.SwitchUseFeedFetchOps(false)
	config.SwitchUseFeedFetchOps(false);
	config.SwitchSpecifyInputNames(true);
	// 开启内存优化
	config.EnableMemoryOptim();
    auto predictor = CreatePaddlePredictor(config);
    // 准备输入tensor
    auto input_names = predictor->GetInputNames();
    auto in_tensor = predictor->GetInputTensor(input_names[0]);
    in_tensor->Reshape(input_shape);
    in_tensor->copy_from_cpu(input_data.data());
    // 运行预测
    predictor->ZeroCopyRun();
    // 处理输出tensor
    auto output_names = predictor->GetOutputNames();
    auto out_tensor = predictor->GetOutputTensor(output_names[output_id]);
    std::vector<int> output_shape = out_tensor->shape();
    // 计算输出的 buffer 大小
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
        output_size *= output_shape[j];
    }
    output_data.resize(output_size);
    out_tensor->copy_to_cpu(output_data.data());
    if (lod_data != nullptr) {
        *lod_data = out_tensor->lod();
    }
}

// 是否带口罩分类模型后处理, 保存分类结果为: (类型, 分数) 对
std::vector<std::pair<int, float>> postprocess_classify(
        const std::vector<float>& output_data,
        int batch_size) {
    // 用于保存每个图的分类结果
    std::vector<std::pair<int, float>> result;
    // 获取数据地址
    auto data = output_data.data();
    auto out_num = output_data.size();
    for (int i = 0; i < batch_size; ++i) {
	    auto out_addr = data + (out_num / batch_size) * i;
	    int best_class_id = 0;
        float best_class_score = *(best_class_id + out_addr);
	    for (int j = 0; j < (out_num / batch_size); ++j) {
            auto infer_class = j;
            auto score = *(j + out_addr);
            // printf("image[%d]: class=%d, score=%.5f\n", i, infer_class, score);
	        if(score > best_class_score) {
		        best_class_id = infer_class;
                best_class_score = score;
	        }
	    }
        // printf("image[%d] : best_class_id=%d, score=%.5f\n", i, best_class_id, best_class_score);
        result.push_back({best_class_id, best_class_score});
    }
    return result;
}

// 人脸检测模型的后处理
std::vector<DetectionOut> postprocess_detection(
        std::vector<float>& output_data, 
        std::vector<std::vector<size_t>>& lod_data,
        std::vector<int>& input_shape,
        std::vector<cv::Mat>& input_mat) {
    std::vector<DetectionOut> result;
    auto rh = input_shape[2];
    auto rw = input_shape[3];
    for (int i = 0; i < lod_data[0].size() - 1; ++i) {
        result.emplace_back(DetectionOut(i));
        for (int j = lod_data[0][i]; j < lod_data[0][i+1]; ++j) {
            // 分类
            int class_id = static_cast<int>(round(output_data[0 + j * 6]));
            // 分数
            float score = output_data[1 + j * 6];
            // 左上坐标
            int top_left_x = output_data[2 + j * 6] * rw;
            int top_left_y = output_data[3 + j * 6] * rh;
            // 右下坐标
            int right_bottom_x = output_data[4 + j * 6] * rw;
            int right_bottom_y = output_data[5 + j * 6] * rh;
            int wd = right_bottom_x - top_left_x;
            int hd = right_bottom_y - top_left_y;
            if (score > g_threshold) {
                std::vector<int> rect = {top_left_x, top_left_y, right_bottom_x, right_bottom_y};
                cv::Rect clip_rect = cv::Rect(top_left_x, top_left_y, wd, hd) & cv::Rect(0, 0, rw, rh);;
                cv::Mat roi = input_mat[i](clip_rect);
                result[i].add_rect(rect, roi);
                /*
                printf("image[%d]: rect[%d] = [(%d, %d), (%d, %d)], score = %.5f\n",
                    i, result[i].get_rect_num() - 1,
                    top_left_x, top_left_y, right_bottom_x, right_bottom_y,
                    score
                );
                */
            }
        }
    }
    return result;
}

void predict(std::vector<std::string>& images, std::string model_dir)
{
    // 人脸检测模型
    std::string detect_model_dir = model_dir + "/pyramidbox_lite/";
    // 面部口罩识别分类模型
    std::string classify_model_dir = model_dir + "/mask_detector/";
    // 人脸检测模型开始预测
    int batch_size = images.size();
    std::vector<float> input_data;
    std::vector<float> output_data;
    std::vector<std::vector<size_t>> lod_data;
    std::vector<int> input_shape;
    std::vector<cv::Mat> input_mat;
    // 检测数据的预处理
    preprocess_batch_detection(images, input_data, input_shape, input_mat);
    // 检测模型预测
    run_predict(detect_model_dir, input_data, input_shape, output_data, 0, &lod_data);
    // 检测模型的后处理
    auto det_out = postprocess_detection(output_data, lod_data, input_shape, input_mat);
    
    printf("det ready!!!\n");
    // 获取检测框 mat 列表
    std::vector<cv::Mat> det_mats = det_out[0].get_mats();
    int cls_batch_size = det_mats.size();
    std::vector<float> cls_input_data;
    std::vector<float> cls_output_data;
    std::vector<int> cls_input_shape = {cls_batch_size, 3, 128, 128};
    // 分类预处理
    preprocess_batch_classify(det_mats, cls_input_data, cls_input_shape);
    // 分类预测
    run_predict(classify_model_dir, cls_input_data, cls_input_shape, cls_output_data, 1);
    // 分类后处理
    auto out = postprocess_classify(cls_output_data, cls_batch_size);
    for (int i = 0; i < cls_batch_size; ++i) {
        auto rect = det_out[0].get_rects()[i];
        auto cls = out[i];
        printf("image[%d].rect[%d] = (%d, %d, %d, %d), label_class=%d, confidence=%.3f\n",
            0, i, rect[0], rect[1], rect[2], rect[3], cls.first, cls.second
        );
    }
}

int main(int argc, char** argv)
{
    std::string model_dir = "/root/projects/PaddleMask/models/";
    std::vector<std::string> images = {
        //"/root/projects/PaddleMask/cpp-det/build/images/mask0/mask_input.png"
        "/root/projects/PaddleMask/cpp-det/build/images/mask1/mask0.jpeg"
    };
    predict(images, model_dir);
    return 0;
}
