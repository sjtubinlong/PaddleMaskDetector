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
#include <iostream>
#include <vector>

#include <paddle_inference_api.h>

#include <utils/config_parser.h>

namespace PaddleCvInference {

    using PD_TYPE = paddle::PaddleDType;

    // Get PD_TYPE type size
    inline int get_pd_type_size(PD_TYPE dtype) {
        if (dtype == PD_TYPE::FLOAT32) {
            return sizeof(float);
        } else if (dtype == PD_TYPE::INT64) {
            return sizeof(int64_t);
        } else if (dtype == PD_TYPE::INT32) {
            return sizeof(int32_t);
        } else if (dtype == PD_TYPE::UINT8) {
            return sizeof(uint8_t);
        }
        return sizeof(float);
    }


    //Store Paddle Predictor's Raw input/output data
    class PaddleRawData {
        public:
            // store PD_TYPE in char array
            std::vector<char> _data;
            // shape of PD_TYPE
            std::vector<int> _shape;
            // data type of each element
            PD_TYPE _dtype;

            // get shape of _data
            std::vector<int> get_shape() const {
                return _shape;
            }
            // get data address of _data
            char* get_mutable_data() {
                return _data.data();
            }
            // get PD_TYPE of _data
            PD_TYPE get_data_type() const {
            	return _dtype;
            }
            // resize data 
            char* resize_data(const std::vector<int>& shape,
                              PD_TYPE dtype=PD_TYPE::FLOAT32) {
                _shape = shape;
                _dtype = dtype;
                auto total_size = 1;
                for (int x : _shape) {
                    total_size *= x;
                }
                auto unit_size = get_pd_type_size(dtype);
                _data.resize(total_size * unit_size);
                return _data.data();
            }
    };

    // Prepare Model Config for Paddle Predictor
    inline void PrepareModelConfig(paddle::AnalysisConfig* config, std::string model_dir, bool enable_gpu) {
        config->SetModel(model_dir + "/__model__",
                        model_dir + "/__params__");
        if (enable_gpu) {
            config->EnableUseGpu(100, 0);
        } else {
            config->DisableGpu();
            config->EnableMKLDNN();
            config->SetCpuMathLibraryNumThreads(8);
        }
        // set config->SwitchUseFeedFetchOps(false) for ZeroCopyTensor
        config->SwitchUseFeedFetchOps(false);
        config->SwitchSpecifyInputNames(true);
        // enable memory optimize
        config->EnableMemoryOptim();
    }

    // Run Predictor and Return Raw Result for PostProcessing
    inline void RunPredictor(std::string model_dir,
                             bool enable_gpu,
                             std::vector<PaddleRawData>& inputs,
                             std::vector<PaddleRawData>& outputs) {
        // set config for predictor
        paddle::AnalysisConfig config;
        PrepareModelConfig(&config, model_dir, enable_gpu);
        auto predictor = CreatePaddlePredictor(config);

        // prepare input tensors
        auto in_tensor_names = predictor->GetInputNames();
        for (int i = 0; i < in_tensor_names.size(); ++i) {
            auto tensor_i = predictor->GetInputTensor(in_tensor_names[i]);
            auto shape_i = inputs[i].get_shape();
            float* in_addr_i = reinterpret_cast<float*>(inputs[i].get_mutable_data());
            // set tensor shape and copy data
            tensor_i->Reshape(shape_i);
            tensor_i->copy_from_cpu(in_addr_i);
        }

        // Run Inference
        predictor->ZeroCopyRun();
        // get all output tensor names
        auto out_tensor_names = predictor->GetOutputNames();
        auto out_tensor_num = out_tensor_names.size();
        // copy all output tensor data
        outputs.resize(out_tensor_num);
        for (int i = 0; i < out_tensor_num; ++i) {
            auto tensor_i = predictor->GetOutputTensor(out_tensor_names[i]);
            auto shape_i = tensor_i->shape();
                  
            printf("sz=%d, shape[%d]=[%d, %d, %d]\n", shape_i.size(), i, shape_i[0], shape_i[1], shape_i[2]);
            outputs[i].resize_data(shape_i, tensor_i->type());
            if (tensor_i->type() == PD_TYPE::INT64) {
                auto out_addr_i = reinterpret_cast<int64_t*>(outputs[i].get_mutable_data());
                tensor_i->copy_to_cpu(out_addr_i);
            } else if (tensor_i->type() == PD_TYPE::INT32) {
                auto out_addr_i = reinterpret_cast<int32_t*>(outputs[i].get_mutable_data());
                tensor_i->copy_to_cpu(out_addr_i);
            } else if (tensor_i->type() == PD_TYPE::UINT8) {
                auto out_addr_i = reinterpret_cast<uint8_t*>(outputs[i].get_mutable_data());
                tensor_i->copy_to_cpu(out_addr_i);
            } else {
                auto out_addr_i = reinterpret_cast<float*>(outputs[i].get_mutable_data());
                tensor_i->copy_to_cpu(out_addr_i);
            }
        }
    }
}
