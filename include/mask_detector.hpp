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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

// MaskDetector Result
struct FaceResult {
  // detection result: face rectangle
  int rect[4];
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
    cv::Mat& im,
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
                     std::vector<FaceResult>& results,
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
  void Preprocess();
  // postprocess result
  void Postprocess();
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_buffer_;
  std::vector<float> input_shape_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  int threshold_;
  float shrink_;
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
  void Preprocess();

  void Postprocess();

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<float> input_shape_;
  std::vector<float> output_data_;
  const std::vector<int> EVAL_CROP_SIZE_ = {3, 128, 128};
  std::vector<float> mean_;
  std::vector<float> scale_;
  float threshold_;
};

inline void FaceDetector::Preprocess() {

};

inline void FaceDetector::Postprocess() {

}

inline void FaceDetector::Predict(
    const cv::Mat& img,
    std::vector<FaceResult>* result,
    float shrink) {

}

inline void MaskClassifier::Preprocess() {

}

inline void MaskClassifier::Postprocess() {

}

inline void MaskClassifier::Predict(std::vector<FaceResult>* faces) {
  
}