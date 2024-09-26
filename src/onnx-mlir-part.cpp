/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ onnx-mlir.cpp - Compiler Driver  ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
// Main function for onnx-mlir-pt with partitioning functionalities.
// Implements main for onnx-mlir-pt driver.
//===----------------------------------------------------------------------===//

#include <iostream>
#include <regex>

#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "src/Builder/Partition.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include "llvm/Support/Debug.h"

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"


#define DEBUG_TYPE "onnx_mlir_main"

using namespace onnx_mlir;

int main(int argc, char *argv[]) {

  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();

  llvm::cl::SetVersionPrinter(getVersionPrinter);

  // Remove unrelated options except common ones and the onnx-mlir options
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  initCompilerConfig();

  std::unique_ptr<Partition> partition = std::make_unique<Partition>();
//  std::cout << "deviceConfigFile_ = " << deviceConfigFile_ << std::endl;
  partition->loadDeviceConfig(deviceConfigFile_);
//  std::cout << "partitionPlanFile_ = " << partitionPlanFile_ << std::endl;
  partition->loadPartitionPlan(partitionPlanFile_);
//  std::cout << "deviceConfigMap size = " << deviceConfigMap->size() << std::endl;

  // Special handling of outputBaseName to derive output filename.
  // outputBaseName must specify a file, so ignore invalid values
  // such as ".", "..", "./", "/.", etc.
  bool b = false;
  if (outputBaseName == "-" ||
      (b = std::regex_match(
           outputBaseName.substr(outputBaseName.find_last_of("/\\") + 1),
           std::regex("[\\.]*$")))) {
    if (b)
      llvm::errs() << "Invalid -o option value " << outputBaseName
                   << " ignored.\n";
    outputBaseName =
        (inputFilename == "-")
            ? "stdin"
            : inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  size_t partNum = 0;
  if(partitionPlanFile_.empty()) {
    partNum = 1;
  } else {
    partNum = partition->size();
  }

  std::vector<mlir::MLIRContext*> contextList;
  for(size_t i = 0; i < partNum; i++) {
    mlir::MLIRContext *ctxt = new mlir::MLIRContext();
    contextList.push_back(ctxt);

    mlir::registerOpenMPDialectTranslation(*ctxt);
    if (!ctxt->isMultithreadingEnabled()) {
      assert(ctxt->getNumThreads() == 1 && "1 thread if no multithreading");
      LLVM_DEBUG(llvm::dbgs() << "multithreading is disabled\n");
    }
    loadDialects(*ctxt);
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> partModules;
  std::string errorMessage;
  int rc;
  auto partitionP = partition.get();

    rc = processInputFileWithPartition(inputFilename, contextList, partModules,
                                       &errorMessage, std::move(partition));


    if (rc != 0) {
    if (!errorMessage.empty())
      llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::cout << "=== compileModule ===" << std::endl;
  int idx = 0;
  std::vector<std::string> funcNameList, outputNameList;
  for(mlir::OwningOpRef<mlir::ModuleOp>& m: partModules) {
    std::string baseName = outputBaseName + "_p" + std::to_string(idx);
//    std::cout << "base name = " << baseName << std::endl;

    emissionTarget = getEmitType(partitionP->getPartitionPlanInfo().partitionEmitMap_[partitionP->getPartitionPlanInfo().partitionNames_[idx]]);
    std::cout << "emit name = " << partitionP->getPartitionPlanInfo().partitionEmitMap_[partitionP->getPartitionPlanInfo().partitionNames_[idx]] << std::endl;
    auto contextP = contextList.at(idx);
    compileModule(m, *contextP, baseName, emissionTarget);

    if(emissionTarget == EmissionTargetType::EmitObj) {

        std::string funcName = "run_p" + std::to_string(idx) + "_" + outputBaseName; // Modify! ==> find from moudles.
        std::cout << "funcName = " << funcName << std::endl;
        funcNameList.push_back(funcName);

        std::string outputName = outputBaseName + "_p" + std::to_string(idx); // Modify! ==> find from moudles.
        std::cout << "outputName = " << outputName << std::endl;
        outputNameList.push_back(outputName);
    }

    idx++;
  }
  if(emissionTarget == EmissionTargetType::EmitObj) {
    partition->genUserMainFile(funcNameList, outputNameList);
  }

  std::cout << "=== end onnx-mlir-pt ===" << std::endl;
  return 0;
}
