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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

// MaskDetector Result
struct FaceResult {
  // detection result: face rectangle
  std::vector<int> rect;
  // detection result: cv::Mat of face rectange
  cv::Mat roi_rect;
  // classification result: confidence
  float score;
  // classification result : class id
  int class_id;
};

// Normalize the image by (pix - mean) * scale
void NormalizeImage(
    const std::vector<float> &mean,
    const std::vector<float> &scale,
    cv::Mat& im, // NOLINT
    float* input_buffer) {
  int height = im.rows;
  int width = im.cols;
  int stride = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base = h * width + w;
      input_buffer[base + 0 * stride] =
          (im.at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      input_buffer[base + 1 * stride] =
          (im.at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      input_buffer[base + 2 * stride] =
          (im.at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

// Visualiztion MaskDetector results
inline void VisualizeResult(const cv::Mat& img,
                     const std::vector<FaceResult>& results,
                     cv::Mat* vis_img) {
}

// Load Model and return model predictor
inline void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    std::unique_ptr<paddle::PaddlePredictor>* predictor) {
  // config the model info
  paddle::AnalysisConfig config;
  config.SetModel(model_dir + "/__model__",
                  model_dir + "/__params__");
  if (use_gpu) {
      config.EnableUseGpu(100, 0);
  } else {
      config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // memory optimization
  config.EnableMemoryOptim();
  *predictor = std::move(CreatePaddlePredictor(config));
}

class FaceDetector {
 public:
  explicit FaceDetector(const std::string& model_dir,
                        const std::vector<float>& mean,
                        const std::vector<float>& scale,
                        bool use_gpu = false,
                        float threshold = 0.7) :
      mean_(mean),
      scale_(scale),
      threshold_(threshold) {
    LoadModel(model_dir, use_gpu, &predictor_);
  }

  // run predictor
  void Predict(
      const cv::Mat& img,
      std::vector<FaceResult>* result,
      float shrink);

 private:
  // preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat, float shrink);
  // postprocess result
  void Postprocess(
      const cv::Mat& raw_mat,
      float shrink,
      std::vector<FaceResult>* result);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<float> output_data_;
  std::vector<int> input_shape_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  float threshold_;
};

class MaskClassifier {
 public:
  explicit MaskClassifier(const std::string& model_dir,
                      const std::vector<float>& mean,
                      const std::vector<float>& scale,
                      bool use_gpu = false,
                      float threshold = 0.5) :
  mean_(mean),
  scale_(scale),
  threshold_(threshold) {
    LoadModel(model_dir, use_gpu, &predictor_);
  }

  void Predict(std::vector<FaceResult>* faces);

 private:
  void Preprocess(std::vector<FaceResult>* faces);

  void Postprocess(std::vector<FaceResult>* faces);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<int> input_shape_;
  std::vector<float> output_data_;
  const std::vector<int> EVAL_CROP_SIZE_ = {3, 128, 128};
  std::vector<float> mean_;
  std::vector<float> scale_;
  float threshold_;
};

inline void FaceDetector::Preprocess(const cv::Mat& image_mat, float shrink) {
  // clone the image : keep the original mat for postprocess
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(), shrink, shrink, cv::INTER_CUBIC);
  im.convertTo(im, CV_32FC3, 1.0);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  input_shape_ = {1, rc, rh, rw};
  input_data_.resize(1 * rc * rh * rw);
  float* buffer = input_data_.data();
  NormalizeImage(mean_, scale_, im, input_data_.data());
}

inline void FaceDetector::Postprocess(
    const cv::Mat& raw_mat,
    float shrink,
    std::vector<FaceResult>* result) {
  result->clear();
  int rect_num = 0;
  int rh = input_shape_[2];
  int rw = input_shape_[3];
  int total_size = output_data_.size() / 6;
  for (int j = 0; j < total_size; ++j) {
    // class id
    int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
    // confidence score
    float score = output_data_[1 + j * 6];
    int xmin = (output_data_[2 + j * 6] * rw) / shrink;
    int ymin = (output_data_[3 + j * 6] * rh) / shrink;
    int xmax = (output_data_[4 + j * 6] * rw) / shrink;
    int ymax = (output_data_[5 + j * 6] * rh) / shrink;
    int wd = xmax - xmin;
    int hd = ymax - ymin;
    if (score > threshold_) {
      auto roi = cv::Rect(xmin, ymin, wd, hd) &
                  cv::Rect(0, 0, rw / shrink, rh / shrink);
      // a view ref to original mat
      cv::Mat roi_ref(raw_mat, roi);
      FaceResult result_item;
      result_item.rect = {xmin, xmax, ymin, ymax};
      result_item.roi_rect = roi_ref;
      result->push_back(result_item);
      std::string name = "roi_" + std::to_string(rect_num) + ".jpeg";
      cv::imwrite(name, roi_ref);
      printf("rect[%d] = {left=%d, right=%d, top=%d, bottom=%d}"
              ", score = %.5f\n",
              rect_num++,
              xmin, xmax, ymin, ymax,
              score);
    }
  }
}

inline void FaceDetector::Predict(const cv::Mat& im,
                                  std::vector<FaceResult>* result,
                                  float shrink) {
  // Preprocess image
  Preprocess(im, shrink);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // run predictor
  predictor_->ZeroCopyRun();
  // get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = out_tensor->shape();
  // calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  // Postprocessing result
  Postprocess(im, shrink, result);
}

inline void MaskClassifier::Preprocess(std::vector<FaceResult>* faces) {
  int batch_size = faces->size();
  input_shape_ = {
      batch_size,
      EVAL_CROP_SIZE_[0],
      EVAL_CROP_SIZE_[1],
      EVAL_CROP_SIZE_[2]
  };
  // reallocate input buffer
  int input_size = 1;
  for (int x : input_shape_) {
    input_size *= x;
  }
  input_data_.resize(input_size);
  auto buffer_base = input_data_.data();
  for (int i = 0; i < batch_size; ++i) {
    cv::Mat im = (*faces)[i].roi_rect;
    // resize
    int rc = im.channels();
    int rw = im.cols;
    int rh = im.rows;
    cv::Size resize_size(input_shape_[3], input_shape_[2]);
    if (rw != input_shape_[3] || rh != input_shape_[2]) {
      cv::resize(im, im, resize_size, 0.f, 0.f, cv::INTER_CUBIC);
    }
    im.convertTo(im, CV_32FC3, 1.0 / 256.0);
    rc = im.channels();
    rw = im.cols;
    rh = im.rows;
    float* buffer_i = buffer_base + i * rc * rw * rh;
    NormalizeImage(mean_, scale_, im, buffer_i);
  }
}

inline void MaskClassifier::Postprocess(std::vector<FaceResult>* faces) {
  float* data = output_data_.data();
  int batch_size = faces->size();
  int out_num = output_data_.size();
  for (int i = 0; i < batch_size; ++i) {
    auto out_addr = data + (out_num / batch_size) * i;
    int best_class_id = 0;
    float best_class_score = *(best_class_id + out_addr);
    for (int j = 0; j < (out_num / batch_size); ++j) {
      auto infer_class = j;
      auto score = *(j + out_addr);
      if (score > best_class_score) {
        best_class_id = infer_class;
        best_class_score = score;
      }
    }
    (*faces)[i].class_id = best_class_id;
    (*faces)[i].score = best_class_score;
  }
}

inline void MaskClassifier::Predict(std::vector<FaceResult>* faces) {
  Preprocess(faces);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // run predictor
  predictor_->ZeroCopyRun();
  // get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[1]);
  std::vector<int> output_shape = out_tensor->shape();
  // calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  Postprocess(faces);
}
