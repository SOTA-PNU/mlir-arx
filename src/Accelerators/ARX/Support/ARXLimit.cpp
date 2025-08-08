/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- NNPALimit.cpp --------------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// The NNPA constant values.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/ARX/Support/ARXLimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <assert.h>
#include <string>

//===----------------------------------------------------------------------===//
// Compatibility checks

/// Convert the input NNPA level, ie. "z16", to a integer value representing the
/// level, ie. "16". When unkown / out of bounds, returns 0.
int64_t convertARXLevel(std::string inputNNPALevel) {
  if (inputNNPALevel.size() != 3 || inputNNPALevel[0] != 'z')
    return 0;
  if (inputNNPALevel[1] == '1') {
    if (inputNNPALevel[2] == '6')
      return 16;
  }
  return 0;
}

/// A function to check whether the input ARX level, ie. "z16", is compatible
/// with the current ARX level.
bool isCompatibleWithARXLevel(std::string inputARXLevel) {
  int64_t inLevel = convertARXLevel(inputARXLevel);
  int64_t mcpuLevel = convertARXLevel(onnx_mlir::mcpu);
  if (inLevel == 0 && mcpuLevel == 0)
    return false;
  return inLevel <= mcpuLevel;
}

//===----------------------------------------------------------------------===//
// Max dimension checks

// The ARX maximum supported dimension index size value by using
// zdnn_get_arx_max_dim_idx_size() This value depends on HW.
// static constexpr int64_t ARX_Z16_MAXIMUM_DIMENSION_INDEX_SIZE = 32768;

int64_t ARXGetMaxForDim(int64_t dim, int64_t rank) {
  assert(rank >= 0 && "expected positive rank");
  assert(dim >= 0 && dim < rank && "dim outside range [0..rank)");
  if (rank > 4)
    return 0;
  // if (isCompatibleWithARXLevel(ARX_Z16))
    // return ARX_Z16_MAXIMUM_DIMENSION_INDEX_SIZE;
  return 0;
}
