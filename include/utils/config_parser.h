// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

namespace PaddleCvInference {

class ModelConfig {
public:

    ModelConfig() {
        _mean = {0.5, 0.5, 0.5};
        _scale = {1.0, 1.0, 1.0};
        _eval_crop_size = {128, 128};
    }
    
    const std::vector<float>& get_model_mean() const {
    	return _mean;
    }

    const std::vector<float>& get_model_scale() const {
    	return _scale;
    }

    const std::vector<int>& get_eval_crop_size() const {
    	return _eval_crop_size;
    }

private:
    std::vector<float> _mean;
    std::vector<float> _scale;
    std::vector<int> _eval_crop_size;
};

}
