
WITH_GPU=ON

PADDLE_DIR=/root/projects/infer_lib/fluid_inference/
CUDA_LIB=/usr/local/cuda/lib64/
CUDNN_LIB=/usr/local/cuda/lib64/
OPENCV_DIR=/root/projects/opencv3gcc4.8/

rm -rf build
mkdir -p build
cd build

cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF
make clean
make -j12
