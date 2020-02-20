# PaddleHub 口罩人脸识别及分类模型C++部署指南

百度通过 `PaddleHub` 开源了业界首个口罩人脸检测及人类模型，该模型可以有效检测在密集人类区域中携带和未携带口罩的所有人脸，同时判断出是否有佩戴口罩。

开发者可以通过 `PaddleHub` 快速体验模型效果搭建在线服务，还可以通过`C++`集成到`Windows`和`Linux`等不同平台。本文档主要介绍如何通过`C++`把模型预测部署起来。


## 前置依赖

### 一. 安装 `PaddlePaddle` 和 `PaddleHub`
1. `PaddlePaddle`的安装, 请点击[官方安装文档](https://paddlepaddle.org.cn/install/quick)选择合适你安装方式
2. `PaddleHub`的安装
```shell
pip install paddlehub
```

### 二. 从`PaddleHub`导出预测模型
运行以下`Python`代码通过模型导出到指定路径：

```python
import paddlehub as hub
# 加载口罩人脸检测及分类模型
module = hub.Module(name="pyramidbox_lite_mobile_mask")
# 导出预测模型到指定路径
module.processor.save_inference_model("./inference_model")
```

导出后的模型路径结构：
```
inference_model
|
├── mask_detector # 口罩人脸分类模型
|   ├── __model__ # 模型文件
│   └── __param__ # 参数文件
|
└── pyramidbox_lite # 口罩人脸检测模型
    ├── __model__ # 模型文件
    └── __param__ # 参数文件

```

**注意:** 请把`inference_model`目录放到合适的路径，**该目录会在程序运行时供程序加载模型**。


### 三. 准备 `C++` 预测所需依赖库
#### 1. 下载`Paddle C++`预测库
PaddlePaddle C++ 预测库主要分为CPU版本和GPU版本。

其中，GPU 版本支持`CUDA 10.0` 和 `CUDA 9.0`:

以下为各版本C++预测库的下载链接：


|  版本   | 链接  |
|  ----  | ----  |
| CPU+MKL版  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-cpu-avx-mkl/fluid_inference.tgz) |
| CUDA9.0+MKL 版  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz) |
| CUDA10.0+MKL 版 | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz) |

更多可用预测库版本，请点击以下链接下载:[C++预测库下载列表](https://paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/build_and_install_lib_cn.html)


下载并解压, 解压后的 `fluid_inference`目录包含的内容：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 请把解压后的目录放到合适的路径，**该目录路径后续会作为编译依赖**使用。

#### 2. 编译安装 OpenCV

```shell
# 0. 切换到工作目录
cd /root/projects
# 1. 下载OpenCV 3.4.6版本源代码
wget -c https://paddleseg.bj.bcebos.com/inference/opencv-3.4.6.zip
# 2. 解压
unzip opencv-3.4.6.zip && cd opencv-3.4.6
# 3. 创建build目录并编译, 这里安装到/root/projects/opencv3目录
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/root/projects/opencv3 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWITH_IPP=OFF -DBUILD_IPP_IW=OFF -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DCMAKE_INSTALL_LIBDIR=lib64 -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_TIFF=ON -DBUILD_TIFF=ON
make -j4
make install
```

其中参数 `CMAKE_INSTALL_PREFIX` 参数指定了安装路径, 上述操作完成后，`opencv` 被安装在 `/root/projects/opencv3` 目录，**该目录后续作为编译依赖**。

### 四. 编译
#### 1. 下载解压项目代码
```
# 切换到工作目录
cd /root/projects/
# 下载本项目代码
wget -c https://github.com/sjtubinlong/PaddleMaskDetector/archive/develop.zip
# 解压代码目录
unzip develop && mv PaddleMaskDetector-develop PaddleMaskDetector
# 进入项目目录
cd PaddleMaskDetector
```
#### 2. 配置编译脚本
打开文件`linux_build.sh`, 看到以下内容:
```shell
# 是否使用GPU
WITH_GPU=ON
# Paddle 预测库路径，参考3.1
PADDLE_DIR=/root/projects/infer_lib/fluid_inference/
# OpenCV 库路径，参考 3.2
OPENCV_DIR=/root/projects/opencv3gcc4.8/
# CUDA库路径, 仅 WITH_GPU=ON 时设置
CUDA_LIB=/usr/local/cuda/lib64/
# CUDNN库路径，仅 WITH_GPU=ON 且 CUDA_LIB有效时设置
CUDNN_LIB=/usr/local/cuda/lib64/

cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF
make -j4
```

把上述参数根据实际情况做修改后，运行脚本编译程序：
```shell
sh linux_build.sh
```

### 五. 运行和可视化

#### 1. 执行预测

可执行文件有 **2** 个参数，第一个是前面导出的`inference_model`路径，第二个是需要预测的图片路径。

示例:
```shell
./build/main /root/projects/inference_model/ ./images/mask_input.png
```

运行后会在图片相同路径下生成和图片名字相同，后缀为`txt`的结果文件，例如示例结果文件为：`mask_input.png.txt`。


#### 2. 可视化
程序输出的结果直接以文本形式存在`txt`文件中，格式如下：
```
top_left_x top_left_y right_bottom_x right_bottom_y label_class confidence
```
即预测库的左上和右下两个点坐标、预测的类型、置信度。


我们提供了一个参考的可视化脚本，假设继续对`5.1`的输出结果进行可视化：

```shell
./scripts/vis.py ./iamges/mask_input.png ./images/mask_input.png.txt
```

结果示例：
