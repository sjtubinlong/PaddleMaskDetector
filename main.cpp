//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

bool g_enable_gpu = false;
// 检测过滤阈值
float g_threshold = 0.7;
// 用于检测预处理图片
float g_shrink = 0.5;

// 加载图片列表为 cv::Mat 数组
bool ReadImage(const std::vector<std::string>& image_paths,
               std::vector<cv::Mat>& image_mats) {
  for (const auto& path : image_paths) {
    cv::Mat im = cv::imread(path, cv::IMREAD_COLOR);
    if (im.data == nullptr || im.empty()) {
      printf("Fail to open image file : [%s]\n", path.c_str());
      return false;
    }
    image_mats.emplace_back(im);
  }

  return true;
}

// 分类模型预处理
bool PreprocessImageClassify(const std::vector<cv::Mat>& image_mats,
                             const std::vector<float>& mean,
                             const std::vector<float>& scale,
                             std::vector<float>& input_data,
                             std::vector<int>& input_shape) {
  // batch 大小
  int batch_size = image_mats.size();
  // 设置输入的数据 shape
  input_shape = {batch_size, 3, 128, 128};
  // 开辟输入数据的 buffer
  input_data.resize(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]);
  // 存储输入数据的起始地址
  auto buffer_base = input_data.data();
  for (int i = 0; i < batch_size; ++i) {
    cv::Mat im = image_mats[i].clone();
    // resize
    int rc = im.channels();
    int rw = im.cols;
    int rh = im.rows;
    cv::Size resize_size(input_shape[3], input_shape[2]);
    if (rw != input_shape[3] || rh != input_shape[2]) {
      cv::resize(im, im, resize_size, 0.f, 0.f, cv::INTER_CUBIC);
    }
    im.convertTo(im, CV_32FC3, 1.0 / 256.0);
    rc = im.channels();
    rw = im.cols;
    rh = im.rows;
    float* buffer_i = buffer_base + i * rc * rw * rh;
    // Image Normalization: (pix - mean) * scale
    for (int h = 0; h < rh; ++h) {
      auto fptr = im.ptr<float>(h);
      int im_index = 0;
      for (int w = 0; w < rw; ++w) {
        for (int c = 0; c < rc; ++c) {
          int top_index = (c * rh + h) * rw + w;
          float pixel = static_cast<float>(fptr[im_index++]);
          pixel = (pixel - mean[c]) * scale[c];
          buffer_i[top_index] = pixel;
        }
      }
    }
  }
  return true;
}

bool PreprocessImageDetect(cv::Mat& image_mat,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale,
                           std::vector<float>& input_data,
                           std::vector<int>& input_shape,
                           float shrink=1.0) {
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(), shrink, shrink, cv::INTER_CUBIC);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  input_shape = {1, rc, rh, rw};
  input_data.resize(1 * rc * rh * rw);
  float* buffer = input_data.data();
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

// 对图片进行预测
void RunPredict(std::string model_dir,
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
std::vector<std::pair<int, float>> PostprocessClassify(
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


struct DetectionResult {
  std::vector<std::vector<int>> rects;
  std::vector<cv::Mat> rect_mats;

  void add_rect(const std::vector<int>& rect, const cv::Mat& im) {
      rects.push_back(rect);
      rect_mats.push_back(im);
  }
};

// 人脸检测模型的后处理
DetectionResult PostprocessDetection(
        std::vector<float>& output_data,
        std::vector<std::vector<size_t>>& lod_data,
        const std::vector<int>& input_shape,
        cv::Mat& input_mat,
        float shrink) {
    // 记录检测结果
    DetectionResult det_out;
    int rect_num = 0;
    int rh = input_shape[2];
    int rw = input_shape[3];
    for (int j = lod_data[0][0]; j < lod_data[0][1]; ++j) {
        // 分类
        int class_id = static_cast<int>(round(output_data[0 + j * 6]));
        // 分数
        float score = output_data[1 + j * 6];
        // 左上坐标
        int top_left_x = (output_data[2 + j * 6] * rw) / shrink;
        int top_left_y = (output_data[3 + j * 6] * rh) / shrink;
        // 右下坐标
        int right_bottom_x = (output_data[4 + j * 6] * rw) / shrink;
        int right_bottom_y = (output_data[5 + j * 6] * rh) / shrink;
        int wd = right_bottom_x - top_left_x;
        int hd = right_bottom_y - top_left_y;
        if (score > g_threshold) {
            std::vector<int> rect = {
                top_left_x,
                top_left_y,
                right_bottom_x,
                right_bottom_y
            };
            auto roi = cv::Rect(top_left_x, top_left_y, wd, hd) &
                       cv::Rect(0, 0, rw / shrink, rh / shrink);
            cv::Mat roi_ref(input_mat, roi);
            cv::Mat roi_deep = roi_ref.clone();
            det_out.add_rect(rect, roi_deep);
            std::string name = "roi_" + std::to_string(rect_num) + ".jpeg";
            cv::imwrite(name, roi_deep);
            printf("rect[%d] = [(%d, %d), (%d, %d)], score = %.5f\n",
                rect_num++,
                top_left_x, top_left_y, right_bottom_x, right_bottom_y,
                score
            );
        }
    }

    return det_out;
}

void Predict(std::vector<std::string>& images, std::string model_dir) {
  // 人脸检测模型
  std::string detect_model_dir = model_dir + "/pyramidbox_lite/";
  // 面部口罩识别分类模型
  std::string classify_model_dir = model_dir + "/mask_detector/";
  // 人脸检测模型开始预测
  int batch_size = images.size();
  std::vector<float> input_data;
  std::vector<float> output_data;
  std::vector<int> input_shape;
  // 存储原图的cv::Mat对象
  std::vector<cv::Mat> input_mat;
  ReadImage(images, input_mat);
  std::vector<float> mean;
  std::vector<float> scale;
  std::vector<std::vector<size_t>> lod_data;

  mean = {104, 117, 123};
  scale = {0.007843, 0.007843, 0.007843};
  PreprocessImageDetect(input_mat[0], mean, scale, input_data, input_shape, g_shrink);
  RunPredict(detect_model_dir, input_data, input_shape, output_data, 0, &lod_data);
  auto det_out = PostprocessDetection(output_data, lod_data, input_shape, input_mat[0], g_shrink);
  mean = {0.5, 0.5, 0.5};
  scale = {1.0, 1.0, 1.0};
  batch_size = det_out.rect_mats.size();
  PreprocessImageClassify(det_out.rect_mats, mean, scale, input_data, input_shape);
  RunPredict(classify_model_dir, input_data, input_shape, output_data, 1);
  auto cls_out = PostprocessClassify(output_data, batch_size);
  for (int i = 0; i < cls_out.size(); ++i) {
     auto rect = det_out.rects[i];
     auto cls = cls_out[i];
     printf("rect[%d] = (%d, %d, %d, %d), label_class=%d, confidence=%.3f\n",
            i, rect[0], rect[1], rect[2], rect[3], cls.first, cls.second);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
      printf("Usage: ./main /path/of/model/dir/ /path/of/input/image\n");
      return -1;
  }
  std::string model_dir = argv[1];
  std::vector<std::string> images = {argv[2]};
  Predict(images, model_dir);
  return 0;
}
