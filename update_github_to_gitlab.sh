#!/bin/bash

git remote add upstream https://github.com/onnx/onnx-mlir
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
git add .
git commit -m "update: onnx-mlir and submodule library"
git push