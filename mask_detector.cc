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

#include "mask_detector.h" // NOLINT

int main(int argc, char* argv[]) {
  bool use_gpu = false;
  image_path = "./images/1.jepg"
  det_model_dir = "./models/pyramidbox_lite"
  cls_model_dir = "./models/mask_detector"

  // Model initialization
  FaceDetector detector(
      det_model_dir,
      mean = {104, 177, 123},
      scale = {0.007843, 0.007843, 0.007843},
      use_gpu,
      threshold = 0.7);

  MaskClassifier classifier(
      cls_model_dir,
      mean = {0.5, 0.5, 0.5},
      scale = {1.0, 1.0, 1.0},
      use_gpu,
      threshold = 0.5);

  // Load image
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);

  std::vector<FaceResult> results;
  // Stage1: Face detection
  // TODO(longbin): we can add clock here
  detector.Predict(img, &results, shrink = 0.7);
  // Stage2: Mask wearing classification
  // TODO(longbin): we can add clock here
  classifier.Predict(&results);

  // Visualization results
  cv::Mat vis_img;
  Visualization(img, results, &vis_img)

  cv::imwrite("result.jpg", vis_img);
}
