/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacementHeuristic.hpp - Place ops using model  -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains heuristics to place operations on CPU or ARX.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/ARX/Conversion/ONNXToHARX/DevicePlacementHeuristic.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/ONNXToHARXCommon.hpp"
#include "src/Accelerators/ARX/Conversion/ONNXToHARX/PerfModel.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <cmath>
#include <functional>

#define DEBUG_TYPE "device-placement-heuristic"
#define DEBUG 2

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Support to classify ops.

bool isMappedToDevice(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && !device.getValue().empty();
}

bool isMappedToCPU(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && device.getValue().equals_insensitive(CPU_DEVICE);
}

bool isMappedToARX(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && device.getValue().equals_insensitive(ARX_DEVICE);
}

// Determine if op is unsuitable because its not an ONNX op of interest, or it
// is already mapped to the CPU device.
bool isARXFriendlyOp(Operation *op) {
  if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
    return false;
  // These ops are ARX unfriendly. Constants are friendly.
  if (isa<ONNXEntryPointOp, ONNXReturnOp>(op))
    return false;
  // If `device` is already set to CPU, it is ARX unfriendly
  if (isMappedToCPU(op))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Support functions op assignment.

// Return true with a debug message reporting reason for success on ARX.
inline bool fasterOnARX(Operation *op, bool significant = false) {
  LLVM_DEBUG({
    if (significant)
      llvm::dbgs() << "  Significantly faster ";
    else
      llvm::dbgs() << "  Faster ";
    llvm::dbgs() << "on ARX model for op:";
    op->dump();
  });
  return true;
}

// Return false with a debug message reporting reason for failure on ARX.
inline bool fasterOnCPU(Operation *op, bool significant = false) {
  LLVM_DEBUG({
    if (significant)
      llvm::dbgs() << "  Significantly faster ";
    else
      llvm::dbgs() << "  Faster ";
    llvm::dbgs() << "on CPU model for op:";
    op->dump();
  });
  return false;
}

inline void assignToARX(Operation *op, MLIRContext *context) {
  LLVM_DEBUG({
    llvm::dbgs() << "Assign to ARX:";
    op->dump();
  });
  op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, ARX_DEVICE));
}

inline void assignToCPU(Operation *op, MLIRContext *context) {
  LLVM_DEBUG({
    llvm::dbgs() << "Assign to CPU:";
    op->dump();
  });
  op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, CPU_DEVICE));
}

//===----------------------------------------------------------------------===//
// Support functions simple cost model analysis, based solely on one operation.

// Simply determine if operation is faster on CPU or ARX.
bool isOpFasterOnARX(Operation *op, const DimAnalysis *dimAnalysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "\nTest cost-benefit of CPU/ARX for op\n";
    op->dump();
  });
  // Estimate time
  double cpuTime, arxTime;
  if (!estimateTimeForOpWithModel(op, dimAnalysis, cpuTime, arxTime)) {
    // No performance model for this operation, assume faster on ARX;
    cpuTime = 1;
    arxTime = 0;
  }
  if (arxTime < cpuTime)
    return fasterOnARX(op);
  return fasterOnCPU(op);
}

//===----------------------------------------------------------------------===//
// Support functions cost/benefit operation that takes stick/unstick into
// account.

struct DevicePlacementWithStickUnstickCost {
  DevicePlacementWithStickUnstickCost() = delete;
  DevicePlacementWithStickUnstickCost(MLIRContext *context, ModuleOp module,
      const DimAnalysis *dimAnalysis, const OpSetType &cpuOps)
      : context(context), dimAnalysis(dimAnalysis), cpuOps(cpuOps) {
    characterizeOps(module);
  }

  // Data
  MLIRContext *context;
  const DimAnalysis *dimAnalysis;
  // All ops that must execute on CPU, aka not eligible to run on  ARX. Ops
  // in this set can be marked as device=CPU.
  const OpSetType &cpuOps;
  // All ops that may execute on ARX. Ops in this set can be marked as
  // device=CPU or ARX.
  OpSetType arxCandidateOps;
  // All ops that run on CPU but do not require stick/unstick at runtime. Ops in
  // thi set can be marked as device=CPU.
  OpSetType arxNeutralOps;

  void characterizeOps(ModuleOp module) {
    arxCandidateOps.clear();
    arxNeutralOps.clear();
    module.walk([&](Operation *op) -> WalkResult {
      // Skip ops that are ARX unfriendly such as ops already assigned to CPU.
      if (!isARXFriendlyOp(op))
        return WalkResult::advance();
      // Ops that cannot/may not go on ARX but can operate on ARX data "for
      // free" are included here in ARX neutral ops.
      // I assume here (not really true) that transpose and reshape can carry
      // the stickified data.
      if (isa<ONNXConstantOp, ONNXTransposeOp, ONNXReshapeOp>(op)) {
        arxNeutralOps.insert(op);
        return WalkResult::advance();
      }
      // Skip ops that the compiler determined are not suitable for ARX.
      if (cpuOps.contains(op))
        return WalkResult::advance();
      // Remaining ops can be mapped to ARX.
      arxCandidateOps.insert(op);
      return WalkResult::advance();
    });
#if DEBUG >= 2
    LLVM_DEBUG({
      llvm::dbgs() << "\nCPU Ops:\n";
      for (auto op : cpuOps) {
        if (isa<ONNXConstantOp, func::FuncOp>(op))
          continue;
        llvm::dbgs() << "cpu ";
        op->dump();
      }
      llvm::dbgs() << "\nARX Neutral Ops:\n";
      for (auto op : arxNeutralOps) {
        if (isa<ONNXConstantOp, func::FuncOp>(op))
          continue;
        llvm::dbgs() << "neutral ";
        op->dump();
      }
      llvm::dbgs() << "\nARX Candidate Ops:\n";
      for (auto op : arxCandidateOps) {
        llvm::dbgs() << "candidate ";
        op->dump();
      }
    });
#endif
  }

  void classifyValueUsage(Value value, Operation *opToSkip, int64_t &cpuOpCount,
      int64_t &arxOpCount, int64_t &arxCandidateOpCount,
      int64_t &arxNeutralOpCount) {
    cpuOpCount = arxOpCount = arxCandidateOpCount = arxNeutralOpCount = 0;

    std::string msg = "";
    for (Operation *userOp : value.getUsers()) {
      // Skip op if requested.
      if (userOp == opToSkip) {
        LLVM_DEBUG(msg = " Skipped op.");
        // Test ops that are already mapped.
      } else if (isMappedToCPU(userOp))
        cpuOpCount++;
      else if (isMappedToARX(userOp))
        arxOpCount++;
      // Not mapped, test now ops that are candidate to execute on ARX.
      else if (arxCandidateOps.contains(userOp))
        arxCandidateOpCount++;
      // Not candidate, test now ops that are neutral to ARX.
      else if (arxNeutralOps.contains(userOp))
        arxNeutralOpCount++;
      // None of the above, will be on CPU.
      else
        cpuOpCount++;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "    Use pattern for value from "
                   << value.getDefiningOp()->getName() << ": used by CPU "
                   << cpuOpCount << ", ARX " << arxOpCount
                   << ", ARX candidates " << arxCandidateOpCount
                   << ", neutral " << arxNeutralOpCount << "." << msg << "\n";
    });
  }

  // Cost benefit analysis of moving this op X to the ARX, with respect the ops
  // that are using the results of op X. Positive cost are additional cost to
  // have op X on ARX, negative costs are benefits to have op X on ARX.
  double costBenefitIncurredForResults(Operation *opX) {
    assert(!isMappedToDevice(opX) && "cannot evaluate an already mapped op");
    double totalCostBenefit = 0;
    LLVM_DEBUG(llvm::dbgs() << "  Look at cost benefit for results:\n");
    for (Value resVal : opX->getResults()) {
      // Look at all the users of currRes and classify them.
      int64_t cpuOpCount, arxOpCount, arxCandidateOpCount, arxNeutralOpCount;
      classifyValueUsage(resVal, /*skip op*/ nullptr, cpuOpCount, arxOpCount,
          arxCandidateOpCount, arxNeutralOpCount);
      /*
        Case study:
        1)  Op X remains on CPU  | 2) Op X migrates to ARX:
                   X.CPU         |          X.ARX
                /    |    \      |      /     |      \
               /   stick? stick  | unstick unstick?   \
              /      |       \   |    /       |        \
            CPU  Candidate  ARX |  CPU   Candidate    ARX
                 on ARX         |        on CPU
        placing X on ARX:       |
            cost:                | +1 unstick if has CPU users
            benefit:             | -1 stick if has ARX users

        TODO: If migrate X to ARX, could attribute some benefits for having
        users that are ARX.
      */
      double costOfUnstickOp = estimateTimeForUnstickOp(resVal);
      double costOfStickOp = estimateTimeForStickOp(resVal);
      if (cpuOpCount > 0) {
        // Moving this op to ARX will cost one unstick as there are one or
        // more ops that must execute on CPU.
        LLVM_DEBUG(
            llvm::dbgs() << "      +1 unstick: " << costOfUnstickOp << "\n");
        totalCostBenefit += costOfUnstickOp;
      }
      if (arxOpCount > 0) {
        // Moving this op to ARX will remove the need to stick this result
        LLVM_DEBUG(
            llvm::dbgs() << "      -1 stick: " << -costOfStickOp << "\n");
        totalCostBenefit -= costOfStickOp;
      }
    }
    return totalCostBenefit;
  }

  // Cost benefit analysis of moving this op X to the ARX, with respect the ops
  // that define the inputs of op X. Positive cost are additional cost to
  // have op X on ARX, negative costs are benefits to have op X on ARX.
  double costBenefitIncurredForInputs(Operation *opX) {
    assert(!isMappedToDevice(opX) && "cannot evaluate an already mapped op");
    double totalCostBenefit = 0;
    LLVM_DEBUG(llvm::dbgs() << "  Look at cost benefit for inputs:\n");
    OpSetType visitedDefiningOps;
    for (Value inputVal : opX->getOperands()) {
      // Investigate the operation that defines inputVal (which is used by op)
      Operation *definingOp = inputVal.getDefiningOp();
      if (!definingOp)
        continue;
      // If we have AddOp(%3, %3), should visit cost associated with %3 input
      // only once.
      if (visitedDefiningOps.contains(definingOp)) {
        LLVM_DEBUG(llvm::dbgs() << "    has multiple use of same input\n");
        continue;
      }
      visitedDefiningOps.insert(definingOp);

      // Classify all other users of this input value.
      int64_t cpuOpCount, arxOpCount, arxCandidateOpCount, arxNeutralOpCount;
      classifyValueUsage(inputVal, /*skip op X that we are analyzing*/ opX,
          cpuOpCount, arxOpCount, arxCandidateOpCount, arxNeutralOpCount);
      /*
        Case study:
        3) Op X remains on CPU           | 4) Op X remains on CPU
                  def.CPU ----.          |        def.ARX -----.
                /    |    \     \        |      /     |    \     \
               /   stick? stick  \       | unstick unstick? \   unstick
              /      |       \    \      |    /       |      \     \
            CPU  Candidate  ARX  X.CPU  |  CPU  Candidate  ARX  X.CPU
                 on ARX                 |       on CPU

        5) Op X migrates to ARX         | 6) Op X migrates to ARX
                  def.CPU ----.          |        def.ARX -----.
                /    |    \     \        |      /     |    \     \
               /   stick? stick stick    | unstick unstick? \     \
              /      |       \    \      |    /       |      \     \
            CPU  Candidate  ARX  X.ARX |  CPU  Candidate  ARX  X.ARX
                 on ARX                 |       on CPU

        placing X on ARX:               |
            cost: +1 stick if first ARX |
            benefit:                     | -1 stick
      */
      double costOfStickOp = estimateTimeForStickOp(inputVal);
      if (isMappedToCPU(definingOp) ||
          !(arxCandidateOps.contains(definingOp) ||
              arxNeutralOps.contains(definingOp))) {
        // Case 5.
        if (arxOpCount == 0) {
          LLVM_DEBUG(llvm::dbgs() << "      def-op on cpu (case 5), +1 stick "
                                  << costOfStickOp << ".\n");
          totalCostBenefit += costOfStickOp;
        }
      }
      if (isMappedToARX(definingOp)) {
        // Case 6.
        LLVM_DEBUG(llvm::dbgs() << "      def-op on ARX (case 6), -1 stick "
                                << -costOfStickOp << ".\n");
        totalCostBenefit -= costOfStickOp;
      }
    }
    return totalCostBenefit;
  }

  bool significantlyFaster(double fast, double slow, double factor) {
    // At least factor x faster.
    return factor * fast <= slow;
  }

  // Determine if op is faster on the ARX or not. To be faster than the CPU,
  // expect the ARX to be at least minFactor faster than CPU. Significant is
  // set if the op is significantFactor faster / slower on the device.
  bool isOpFasterOnARX(Operation *op, double minFactor,
      double significantCPUFactor, double significantARXFactor,
      bool &significant) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "\nTest cost-benefit with stick/unstick of CPU/ARX for op\n";
      op->dump();
    });
    // Estimate time
    double cpuTime, arxTime, arxTimeWithOverheads;
    if (estimateTimeForOpWithModel(op, dimAnalysis, cpuTime, arxTime)) {
      // Has performance model, account for stick/unstick.
      double useCostBenefit = costBenefitIncurredForResults(op);
      double inputCostBenefit = costBenefitIncurredForInputs(op);
      arxTimeWithOverheads = arxTime + useCostBenefit + inputCostBenefit;
      LLVM_DEBUG(llvm::dbgs()
                 << "  New estimated arx time with stick/unstick:"
                 << arxTimeWithOverheads << " vs cpu " << cpuTime << ".\n");
    } else {
      // No performance model for this operation, assume faster on ARX;
      cpuTime = 10;
      arxTime = arxTimeWithOverheads = 1;
      LLVM_DEBUG(llvm::dbgs() << "    no time estimate, assume ARX better\n.");
    }
    if (arxTimeWithOverheads * minFactor <= cpuTime) {
      // For significant, don't take overheads into account as it may change
      // depending on mapping.
      significant =
          significantlyFaster(arxTime, cpuTime, significantARXFactor);
      return fasterOnARX(op, significant);
    }
    // For significant, don't take overheads into account as it may change
    // depending on mapping.
    significant = significantlyFaster(cpuTime, arxTime, significantCPUFactor);
    return fasterOnCPU(op, significant);
  }

}; // DevicePlacementWithStickUnstickCost

} // namespace

//===----------------------------------------------------------------------===//
// Exported heuristics for device placement.

namespace onnx_mlir {

void PlaceAllLegalOpsOnARX(MLIRContext *context,
    const SmallVector<Operation *, 32> &ops, const OpSetType &cpuOps) {
  for (Operation *op : ops) {
    if (isMappedToDevice(op))
      continue;
    // Op that cannot go on ARX.
    if (cpuOps.contains(op))
      continue;
    // Compiler determined that we want this op on the ARX, mark as such.
    assignToARX(op, context);
  }
}

void PlaceBeneficialOpsOnARX(MLIRContext *context,
    const SmallVector<Operation *, 32> &ops, const DimAnalysis *dimAnalysis,
    const OpSetType &cpuOps) {
  for (Operation *op : ops) {
    if (isMappedToDevice(op))
      continue;
    // Op that cannot go on ARX.
    if (cpuOps.contains(op))
      continue;
    // Now we have an operation that can work on the ARX, check if its
    // beneficial
    if (!isOpFasterOnARX(op, dimAnalysis)) {
      assignToCPU(op, context);
      continue;
    }
    // Compiler determined that we want this op on the ARX, mark as such.
    assignToARX(op, context);
  }
}

void PlaceBeneficialOpsOnARXWithStickUnstick(MLIRContext *context,
    ModuleOp module, const SmallVector<Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps, double minFactor,
    double significantCPUFactor, double significantARXFactor) {
  // Init model.
  DevicePlacementWithStickUnstickCost model(
      context, module, dimAnalysis, cpuOps);
  int64_t ub = 5;
  int64_t i = 0;
  while (i < ub) {
    int64_t modified = 0;
    bool first = (i == 0);
    bool last = (i == ub - 1);
    LLVM_DEBUG(llvm::dbgs() << "\n\n\nPlacement Iteration " << i << "\n\n");
    for (Operation *op : ops) {
      if (isMappedToDevice(op))
        continue;
      // Op that cannot go on ARX.
      if (cpuOps.contains(op))
        continue;
      // Now we have an operation that can work on the ARX, check if its
      // beneficial
      bool significant;
      if (!model.isOpFasterOnARX(op, minFactor, significantCPUFactor,
              significantARXFactor, significant)) {
        if (last || significant) {
          modified++;
          assignToCPU(op, context);
        }
        continue;
      }
      // Compiler determined that we want this op on the ARX, mark as such.
      if (!first || significant) {
        modified++;
        assignToARX(op, context);
      }
    }
    if (last) {
      break;
    } else if (first) {
      LLVM_DEBUG(llvm::dbgs() << "\nFirst, go on.\n");
      ++i;
    } else if (modified) {
      LLVM_DEBUG(llvm::dbgs() << "\nHad " << modified << " changes, go on.\n");
      ++i;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "\nHad no changes, skip to last iter.\n");
      i = ub - 1;
    }
  }
}

} // namespace onnx_mlir
