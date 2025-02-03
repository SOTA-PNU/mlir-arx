#!/bin/bash

# In: /home/{username}/app/3rdparty

git clone --recursive https://github.com/onnx/onnx-mlir.git

MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi
cmake --build .

export LIT_OPTS=-v
cmake --build . --target check-onnx-lit