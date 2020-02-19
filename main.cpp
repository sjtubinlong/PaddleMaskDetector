#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>

#include <paddle_cv_inference.h>

DEFINE_string(model_dir, "", "Configuration File Path");
DEFINE_string(input_dir, "", "Directory of Input Images");

int main()
{
   std::string model_dir1 = "/root/projects/PaddleMask/models/mask_detector/";
   std::string model_dir2 = "/root/projects/PaddleMask/models/pyramidbox_lite/";
   bool use_gpu = true;
   std::vector<std::string> imgs = {
       "/root/projects/PaddleMask/cpp-det/build/images/mask1/mask0.jpeg",
       "/root/projects/PaddleMask/cpp-det/build/images/mask1/mask0.jpeg"
   };
   
   PaddleCvInference::ModelConfig model_config;

   std::vector<PaddleCvInference::PaddleRawData> inputs(1);
   std::vector<PaddleCvInference::PaddleRawData> outputs;

   inputs[0].resize_data({2, 3, 128, 128});

   preprocess_batch(imgs, inputs[0], model_config);
   PaddleCvInference::RunPredictor(model_dir1, use_gpu, inputs, outputs);   
   postprocess(outputs[1]);

   PaddleCvInference::RunPredictor(model_dir2, use_gpu, inputs, outputs);   
   return 0;
}
