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
  int id;
  cv::Mat rect;
  float score;
  int class_id;
};

// Normalize the image by (pix - mean) * scale
void NormalizeImage(cv::Mat& img, float* img_buf) {
}

// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     std::vector<FaceResult>& results,
                     cv::Mat* vis_img) {
}

class FaceDetector {
 public:
  bool FaceDetector(const std::string& model_dir,
            const std::vector<float>& mean,
            const std::vector<float>& scale,
            bool use_gpu = false,
            threshold = 0.7) :
    mean_(mean),
    scale_(scale) {
  }

  void Predict(const cv::Mat& img, std::vector<FaceResult>* result);

 private:
  void Postprocess();
  void Preprocess();
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  vector<float> mean_;
  std::vector<float> scale_;
};

class MaskClassifier {
 public:
  bool MaskClassifier(const std::string& model_dir,
                      const std::vector<float>& mean,
                      const std::vector<float>& scale,
                      bool use_gpu = false,
                      threshold = 0.5) :
  mean_(mean),
  scale_(scale) {
  }

  void Predict(std::vector<FaceResult>* faces);

 private:
  void Preprocess();

  void Postprocess();

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
