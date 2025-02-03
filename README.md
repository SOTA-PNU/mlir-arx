# ONNX-MLIR



## Getting started
```
$ git clone --recursive https://gitlab.com/m0z3li/mlir-arx.git
$ cd mlir-arx
$ docker build -t onnx-mlir .  
$ docker run -d -p 8888:22 -v $(pwd)/app onnx-mlir
$ ssh root@localhost -p 8888 
# cd /home/{username}/app && ./setup.sh
```