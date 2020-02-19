
#include <string>
#include <vector>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <paddle_inference_api.h>

bool g_enable_gpu = false;
// 检测过滤阈值
float g_threshold = 0.7;

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
    // 添加一个检测框结果
    void add_rect(std::vector<int>& rect) {
        rectange.emplace_back(rect);
    }
};

// 图片预处理: 输入直接为cv::Mat
bool preprocess_image(cv::Mat& im, float* buffer, const std::vector<int>& input_shape) {
    if (im.data == nullptr || im.empty()) {
        printf("Invalid Mat input\n");
        return false;
    }
    // resize
    int channels = im.channels();
	int rw = im.cols;
    int rh = im.rows;
    cv::Size resize_size(input_shape[2], input_shape[3]);
    if (rw != input_shape[2] || rh != input_shape[3]) {
        cv::resize(im, im, resize_size);
    }
    // BGR2RGB
    cvtColor(im, im, CV_BGR2RGB);
    // 减均值除方差: (img - mean) * scale
    int hh = im.rows;
    int ww = im.cols;
    int cc = im.channels();
    float mean[3] = {104, 117, 123};
    float scale[3] = {0.007843, 0.007843, 0.007843};
    #pragma omp parallel for
    for (int h = 0; h < hh; ++h) {
        float* ptr = im.ptr<float>(h);
        int im_index = 0;
        for (int w = 0; w < ww; ++w) {
            for (int c = 0; c < cc; ++c) {
                int top_index = (c * hh + h) * ww + w;
                float pixel = static_cast<float>(ptr[im_index++]);
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
        return preprocess_image(im, buffer, input_shape);
}

// 用于分类的图片批量预处理: 批量处理
template<typename T>
bool preprocess_batch(
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
        auto im = images[i];
        auto base = input_data.data();
        auto buffer = reinterpret_cast<float*>(base + i * item_size);
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

// 对图片进行预测
 void run_predict(std::string model_dir,
            const std::vector<float>& input_data,
            const std::vector<int>& input_shape,
            std::vector<float>& output_data,
            int output_id=0) {
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
            printf("image[%d]: class=%d, score=%.5f\n", i, infer_class, score);
	        if(score > best_class_score) {
		        best_class_id = infer_class;
                best_class_score = score;
	        }
	    }
        printf("image[%d] : best_class_id=%d, score=%.5f\n", i, best_class_id, best_class_score);
        result.push_back({best_class_id, best_class_score});
    }
    return result;
}

// 人脸检测模型的后处理
std::vector<DetectionOut> postprocess_detection(
        std::vector<float>& output_data, 
        const std::vector<size_t>& lod_vector) {
    std::vector<int> resize_widths;
    std::vector<int> resize_heights;
    std::vector<DetectionOut> result;
    for (int i = 0; i < lod_vector.size() - 1; ++i) {
        result.emplace_back(DetectionOut(i));
        for (int j = lod_vector[i]; j < lod_vector[i+1]; ++j) {
            // 分类
            int class_id = static_cast<int>(round(output_data[0 + j * 6]));
            // 分数
            float score = output_data[1 + j * 6];
            // 左上坐标
            int top_left_x = output_data[2 + j * 6] * resize_widths[i];
            int top_left_y = output_data[3 + j * 6] * resize_heights[i];
            // 右下坐标
            int right_bottom_x = output_data[4 + j * 6] * resize_widths[i];
            int right_bottom_y = output_data[5 + j * 6] * resize_heights[i];
            if (score > g_threshold) {
                std::vector<int> rect = {top_left_x, top_left_y, right_bottom_x, right_bottom_y};
                result[i].add_rect(rect);
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

    // 分类模型开始预测
    int batch_size = images.size();
    std::vector<int> input_shape = {batch_size, 3, 128, 128};
    std::vector<float> input_data;
    std::vector<float> output_data;
    // 预处理
    preprocess_batch(images, input_data, input_shape);
    // 预测
    run_predict(classify_model_dir, input_data, input_shape, output_data, 1);
    // 后处理
    auto out = postprocess_classify(output_data, batch_size);
}

int main(int argc, char** argv)
{
    std::string model_dir = "/root/projects/PaddleMask/models/";
    std::vector<std::string> images = {
    	"/root/projects/PaddleMask/cpp-det/build/images/mask1/mask0.jpeg",
        "/root/projects/PaddleMask/cpp-det/build/images/mask0/mask_input.png"
    };
    predict(images, model_dir);
    return 0;
}
