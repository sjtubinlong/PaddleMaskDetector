cmake_minimum_required(VERSION 3.0)
project(PaddleMaskDetector CXX C)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support,defaultuseMKL."          ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)

SET(PADDLE_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WITH_MKL)
    ADD_DEFINITIONS(-DUSE_MKL)
endif()

if (NOT DEFINED PADDLE_DIR OR ${PADDLE_DIR} STREQUAL "")
    message(FATAL_ERROR "please set PADDLE_DIR with -DPADDLE_DIR=/path/paddle_influence_dir")
endif()

if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${PADDLE_DIR}/")
include_directories("${PADDLE_DIR}/third_party/install/protobuf/include")
include_directories("${PADDLE_DIR}/third_party/install/glog/include")
include_directories("${PADDLE_DIR}/third_party/install/gflags/include")
include_directories("${PADDLE_DIR}/third_party/install/xxhash/include")
if (EXISTS "${PADDLE_DIR}/third_party/install/snappy/include")
    include_directories("${PADDLE_DIR}/third_party/install/snappy/include")
endif()
if(EXISTS "${PADDLE_DIR}/third_party/install/snappystream/include")
    include_directories("${PADDLE_DIR}/third_party/install/snappystream/include")
endif()
include_directories("${PADDLE_DIR}/third_party/install/zlib/include")
include_directories("${PADDLE_DIR}/third_party/boost")
include_directories("${PADDLE_DIR}/third_party/eigen3")

if (EXISTS "${PADDLE_DIR}/third_party/install/snappy/lib")
    link_directories("${PADDLE_DIR}/third_party/install/snappy/lib")
endif()
if(EXISTS "${PADDLE_DIR}/third_party/install/snappystream/lib")
    link_directories("${PADDLE_DIR}/third_party/install/snappystream/lib")
endif()

link_directories("${PADDLE_DIR}/third_party/install/zlib/lib")
link_directories("${PADDLE_DIR}/third_party/install/protobuf/lib")
link_directories("${PADDLE_DIR}/third_party/install/glog/lib")
link_directories("${PADDLE_DIR}/third_party/install/gflags/lib")
link_directories("${PADDLE_DIR}/third_party/install/xxhash/lib")
link_directories("${PADDLE_DIR}/paddle/lib/")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")
if (WIN32)
  include_directories("${PADDLE_DIR}/paddle/fluid/inference")
  include_directories("${PADDLE_DIR}/paddle/include")
  link_directories("${PADDLE_DIR}/paddle/fluid/inference")
  include_directories("${OPENCV_DIR}/build/include")
  include_directories("${OPENCV_DIR}/opencv/build/include")
  link_directories("${OPENCV_DIR}/build/x64/vc14/lib")
else ()
  include_directories("${PADDLE_DIR}/paddle/include")
  link_directories("${PADDLE_DIR}/paddle/lib")
  include_directories("${OPENCV_DIR}/include")
  link_directories("${OPENCV_DIR}/lib64")
endif ()

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o2 -fopenmp -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
endif()

# TODO let users define cuda lib path
if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    endif(NOT WIN32)
endif() 


if (NOT WIN32)
  if (USE_TENSORRT AND WITH_GPU)
      include_directories("${PADDLE_DIR}/third_party/install/tensorrt/include")
      link_directories("${PADDLE_DIR}/third_party/install/tensorrt/lib")
  endif()
endif(NOT WIN32)

if (NOT WIN32)
    set(NGRAPH_PATH "${PADDLE_DIR}/third_party/install/ngraph")
    if(EXISTS ${NGRAPH_PATH})
        include(GNUInstallDirs)
        include_directories("${NGRAPH_PATH}/include")
        link_directories("${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}")
        set(NGRAPH_LIB ${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}/libngraph${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
endif()

if(WITH_MKL)
  include_directories("${PADDLE_DIR}/third_party/install/mklml/include")
  if (WIN32)
    set(MATH_LIB ${PADDLE_DIR}/third_party/install/mklml/lib/mklml.lib
            ${PADDLE_DIR}/third_party/install/mklml/lib/libiomp5md.lib)
  else ()
    set(MATH_LIB ${PADDLE_DIR}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${PADDLE_DIR}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    execute_process(COMMAND cp -r ${PADDLE_DIR}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
  endif ()
  set(MKLDNN_PATH "${PADDLE_DIR}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if (WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else ()
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    endif ()
  endif()
else()
  set(MATH_LIB ${PADDLE_DIR}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if (WIN32)
    if(EXISTS "${PADDLE_DIR}/paddle/fluid/inference/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(DEPS
            ${PADDLE_DIR}/paddle/fluid/inference/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
        set(DEPS
            ${PADDLE_DIR}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
endif()

if(WITH_STATIC_LIB)
    set(DEPS
        ${PADDLE_DIR}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(DEPS
        ${PADDLE_DIR}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB} 
        glog gflags protobuf z xxhash
        )
    if(EXISTS "${PADDLE_DIR}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
    if (EXISTS "${PADDLE_DIR}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
else()
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        opencv_world346 glog gflags_static libprotobuf zlibstatic xxhash)
    set(DEPS ${DEPS} libcmt shlwapi)
    if (EXISTS "${PADDLE_DIR}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
    if(EXISTS "${PADDLE_DIR}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${PADDLE_DIR}/third_party/install/tensorrt/lib/libnvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${PADDLE_DIR}/third_party/install/tensorrt/lib/libnvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if (NOT WIN32)
    set(DEPS ${DEPS} ${OPENCV_DIR}/lib64/libopencv_imgcodecs${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/lib64/libopencv_imgproc${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/lib64/libopencv_core${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/lib64/libopencv_highgui${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/libIlmImf${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/liblibjasper${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/liblibpng${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/liblibtiff${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/libittnotify${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/liblibjpeg-turbo${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/liblibwebp${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENCV_DIR}/share/OpenCV/3rdparty/lib64/libzlib${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

add_executable(mask_detector main.cc mask_detector.cc)
target_link_libraries(mask_detector ${DEPS})

if (WIN32)
    add_custom_command(TARGET mask_detector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mkldnn/lib/mkldnn.dll ./mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mklml/lib/mklml.dll ./release/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_DIR}/third_party/install/mkldnn/lib/mkldnn.dll ./release/mkldnn.dll
    )
endif()
