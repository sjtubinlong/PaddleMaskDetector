cmake .. \
    -DWITH_GPU=ON \
    -DPADDLE_DIR=/root/projects/infer_lib/fluid_inference/ \
    -DCUDA_LIB=/usr/local/cuda/lib64/ \
    -DOPENCV_DIR=/root/projects/opencv3gcc4.8/ \
    -DCUDNN_LIB=/usr/local/cuda/lib64/ \
    -DWITH_STATIC_LIB=OFF
make clean
make -j12
