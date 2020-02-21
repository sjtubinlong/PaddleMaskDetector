# PaddleHub口罩人脸识别及分类模型C++预测部署

百度通过 `PaddleHub` 开源了业界首个口罩人脸检测及人类模型，该模型可以有效检测在密集人类区域中携带和未携带口罩的所有人脸，同时判断出是否有佩戴口罩。开发者可以通过 `PaddleHub` 快速体验模型效果、搭建在线服务，还可以导出模型集成到`Windows`和`Linux`等不同平台的`C++`开发项目中。

本文档主要介绍如何把模型在`Windows`和`Linux`上完成基于`C++`的预测部署。

主要包含两个步骤：
- [1. 模型预测模型](#1PaddleHub导出预测模型)
- [2. 编译 C++](#2C++预测部署编译)

## 1.PaddleHub导出预测模型

#### 1.1 安装 `PaddlePaddle` 和 `PaddleHub`

1. `PaddlePaddle`的安装, 请点击[官方安装文档](https://paddlepaddle.org.cn/install/quick)选择合适你安装方式
2. `PaddleHub`的安装
```shell
pip install paddlehub
```

如果**已安装**好，直接跳过本步骤。

#### 1.2 从`PaddleHub`导出预测模型
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
## 2.C++预测部署编译

本项目支持在`Windows`和`Linux`上编译并部署`C++`项目，不同平台的编译请参考：
- [Linux 编译](./docs/linux_build.md)
- [Windows 使用 Visual Studio 2019编译](./docs/windows_build.md)
