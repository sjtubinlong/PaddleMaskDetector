cmake .. \
    -DWITH_GPU=ON \
    -DPADDLE_DIR=/docker/deploy-Dec/fluid_inference/ \
    -DCUDA_LIB=/usr/local/cuda/lib64/ \
    -DOPENCV_DIR=/root/projects/opencv3/ \
    -DCUDNN_LIB=/usr/lib/x86_64-linux-gnu/ \
    -DWITH_STATIC_LIB=OFF
make clean
make -j12
