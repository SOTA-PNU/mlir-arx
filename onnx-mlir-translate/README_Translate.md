# onnx-mlir-translate 


## 컴파일

컴파일시.

CompilerOptions.cpp 파일을 src/Compiler 디렉토리로 복사 시킨다.


25라인을 살펴보면 이부분을 1로 하여야, onnx-mlir-translate가 동작한다.
compiler 옵션중에 5개 정도가, 기존 mlir과 충돌이 하게되기 때문에, 충돌을 방지하기 위해서,
설정하게 된다.

```cpp

#define IS_ONNX_TRANSLATE 1

```

/root/arx/release/onnx-mlir --EmitONNXIR resnet18-v2-7.onnx

## 실행 방법

```sh
onnx-mlir --EmitONNXIR resnet18-v2-7.onnx

onnx-mlir-translate --mlir-to-python  resnet18-v2-7.onnx.mlir > resnet18-v2-7.py
```


```sh
python resnet18-v2-7.py new.onnx
```

