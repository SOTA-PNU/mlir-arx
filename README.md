# ONNX-MLIR



## Getting started
```
$ git clone --recursive https://gitlab.com/m0z3li/mlir-arx.git
$ cd mlir-arx
$ docker build -t onnx-mlir .  
$ docker run --name onnx-mlir -d -v $(pwd)/app onnx-mlir
$ docker exec -it onnx-mlir /bin/bash

# cd /home/{username}/app && ./setup.sh

```