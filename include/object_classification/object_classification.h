#pragma once
#include <vector>
#include <string>
#include <memory>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>

#include <utils/utils.h>

namespace PaddleCvInference {
    // preprocess from a cv::Mat
    inline bool preprocess(cv::Mat& im, float* input_data, const ModelConfig& model_config) {
        if (im.data == nullptr || im.empty()) {
            LOG(ERROR) << "Invalid cv::Mat";
            return false;
        }
        // resize
        auto crop_size = model_config.get_eval_crop_size();
        int channels = im.channels();
	int rw = im.cols;
        int rh = im.rows;
        cv::Size resize_size(crop_size[0], crop_size[1]);
        if (rw != crop_size[0] || rh != crop_size[1]) {
            cv::resize(im, im, resize_size);
        }
        cvtColor(im, im, CV_BGR2RGB);
        // normalize: (img - mean) * scale
        int hh = im.rows;
        int ww = im.cols;
        int cc = im.channels();
        auto mean = model_config.get_model_mean();
        auto scale = model_config.get_model_scale();
        #pragma omp parallel for
        for (int h = 0; h < hh; ++h) {
            float* ptr = im.ptr<float>(h);
            int im_index = 0;
            for (int w = 0; w < ww; ++w) {
                for (int c = 0; c < cc; ++c) {
                    int top_index = (c * hh + h) * ww + w;
                    float pixel = static_cast<float>(ptr[im_index++]);
                    pixel = (pixel - mean[c]) * scale[c];
                    input_data[top_index] = pixel;
                }
            }
        }
        return true;
    }

    // preprocess from a image file path
    inline bool preprocess(std::string fname, float* input_data, const ModelConfig& model_config) {
        cv::Mat im = cv::imread(fname);
        if (im.data == nullptr || im.empty()) {
            LOG(ERROR) << "Failed to open image: [" << fname << "]";
            return false;
        }
        im.convertTo(im, CV_32FC3, 1 / 255.0);
        return preprocess(im, input_data, model_config);
    }

    // preprocess a batch of images (cv::Mat or filepath)
    template<typename T>
    bool preprocess_batch(const std::vector<T>& imgs,
                          PaddleRawData& input,
                          const ModelConfig& model_config) {
        auto shape = input.get_shape();
        auto item_size = shape[0] * shape[1] * shape[2];
        std::vector<std::thread> threads;
        for (int i = 0; i < imgs.size(); ++i) {
            auto im = imgs[i];
            auto base = input.get_mutable_data();
            auto buffer = reinterpret_cast<float*>(base + i * item_size);
            threads.emplace_back([im, buffer, model_config] {
                preprocess(im, buffer, model_config);
                });
        }
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        return true;
    }

    // postprocess output
    std::vector<std::pair<int, float>> postprocess(PaddleRawData& output) {
        // store result as <class, score> pairs
        std::vector<std::pair<int, float>> result;
        // parse the raw output from predictor
        auto shape = output.get_shape();
        auto data = reinterpret_cast<float*>(output.get_mutable_data());
        auto batch_size = shape[0];
        auto out_num = shape[0] * shape[1];
        for (int i = 0; i < batch_size; ++i) {
	    float* out_addr = data + (out_num / batch_size) * i;
	    int max_idx = 0;
	    for (int j = 0; j < (out_num / batch_size); ++j) {
	        std::cout << "img[" << i << "] : class[" << j
                          << "]_score =" << *(j + out_addr) << std::endl;
	        if(*(j + out_addr) > *(max_idx + out_addr)) {
		    max_idx = j;
	        }
	    }
	    std::cout << "class: " << max_idx
                      << "\tscore:" << *(max_idx + out_addr) << std::endl;
            result.push_back({max_idx, *(max_idx + out_addr)});
        }
        return result;
    }
}
