//
// Created by msyu on 24. 3. 4.
//
#pragma once

//#ifndef ONNX_MLIR_PARTITION_H
//#define ONNX_MLIR_PARTITION_H

#include "llvm/ADT/ArrayRef.h"
#include <string>
#include <vector>
#include <map>

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

namespace onnx_mlir {

class PartitionPlanInfo {

public:
  std::string modelName_;
  std::vector<std::string> partitionNames_;
  std::vector<std::string> nodeNames_;
  std::map<std::string, std::string> partitionBackendMap_;
  std::map<std::string, std::string> partitionEmitMap_;
  std::map<std::string, std::string> nodePartitionMap_;
};


class Partition {

private:
  PartitionPlanInfo partitionPlanInfo_;
  std::unique_ptr<std::map<std::string,
      std::unique_ptr<std::map<std::string, std::string>>>>
      deviceConfigMap_ = std::make_unique<std::map<std::string,
          std::unique_ptr<std::map<std::string, std::string>>>>();

public:
  void loadDeviceConfig(std::string deviceConfig);
  void loadPartitionPlan(std::string partitionPlan);
  const int size() { return partitionPlanInfo_.partitionNames_.size();  }
  PartitionPlanInfo getPartitionPlanInfo();
  void genUserMainFile(std::vector<std::string>& funcNameList, std::vector<std::string>& outputNameList);

};

}

//#endif // ONNX_MLIR_PARTITION_H
