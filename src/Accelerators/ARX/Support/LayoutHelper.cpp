/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutHelper.cpp - NNPA Layout Helper ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Support/LayoutHelper.hpp"

using namespace mlir;

namespace onnx_mlir {


bool is2DLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_2D));
}

bool is3DSLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_3DS));
}

bool is4DLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_4D));
}

bool isNHWCLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_NHWC));
}

mlir::StringAttr getNCHWLayoutAttr(PatternRewriter &rewriter) {
  return rewriter.getStringAttr(LAYOUT_NCHW);
}

} // namespace onnx_mlir
