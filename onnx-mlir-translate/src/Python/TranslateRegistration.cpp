//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
//#include "mlir/Dialect/EmitPython/IR/EmitPython.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
//#include "mlir/Target/Python/PythonEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace mlir {


    LogicalResult translateToPython(Operation* op, raw_ostream& os,
            bool declareVariablesAtTop = false,
            StringRef fileId = {});


    void registerToPythonTranslation() {
        static llvm::cl::opt<bool> declareVariablesAtTop(
            "python-declare-variables-at-top",
            llvm::cl::desc("Declare variables at top when emitting python"),
            llvm::cl::init(false));

        static llvm::cl::opt<std::string> fileId(
            "python-file-id", llvm::cl::desc("Emit emitpython.file ops with matching id"),
            llvm::cl::init(""));

        TranslateFromMLIRRegistration reg(
            "mlir-to-python", "translate from mlir to python",
            [](Operation* op, raw_ostream& output) {
                return translateToPython(
                    op, output,
                    /*declareVariablesAtTop=*/declareVariablesAtTop,
                    /*fileId=*/fileId);
            },
            [](DialectRegistry& registry) {

                registry.insert<cf::ControlFlowDialect,
                ONNXDialect,
                func::FuncDialect>();
        // clang-format on
            });
    }

}

