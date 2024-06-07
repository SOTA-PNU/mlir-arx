//
// Created by msyu on 24. 3. 4.
//

#include "Partition.hpp"
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <llvm/Support/FileSystem.h>

using namespace onnx_mlir;

namespace onnx_mlir {
extern std::vector<std::pair<int64_t, std::string>> inOutDims_;
extern std::string emitFolder_;
std::vector<std::string> splitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

PartitionPlanInfo Partition::getPartitionPlanInfo() {
  return partitionPlanInfo_;
}

void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

void Partition::loadPartitionPlan(std::string partitionPlan) {

  if (partitionPlan.empty()) {
//    std::cout << "No partitionPlan file. PartitionPlan.yaml is generated." << std::endl;
    partitionPlanInfo_.partitionNames_.clear();
    return;
  }

  try {

    YAML::Node ppInfo = YAML::LoadFile(partitionPlan);
    if (ppInfo.IsSequence()) {

      std::vector<std::string> bnames, enames;
      std::vector<std::string> nplist;
      std::string input;
      char delimiter = ':';

      for (auto pi: ppInfo) {
        const YAML::Node &node = pi;

        //        std::cout << "Item " << i + 1 << std::endl;
        std::cout << "Name: " << node["name"].as<std::string>() << std::endl;

        partitionPlanInfo_.modelName_ = node["name"].as<std::string>();
        if (node["numPartitions"]) {
          std::cout << "numPartitions: "
                    << (node["numPartitions"].as<int32_t>())
                    << std::endl;
        }
        if (node["backendNames"]) {
          std::cout << "backendNames: "
                    << (node["backendNames"].as<std::string>())
                    << std::endl;

          input = node["backendNames"].as<std::string>();
          bnames = splitString(input, delimiter);

        }

        if (node["emitNames"]) {
          std::cout << "emitdNames: "
                    << (node["emitNames"].as<std::string>())
                    << std::endl;

          input = node["emitNames"].as<std::string>();
          enames = splitString(input, delimiter);

        }

        if (node["partitionNames"]) {
          std::cout << "partitionNames: "
                    << (node["partitionNames"].as<std::string>())
                    << std::endl;

          input = node["partitionNames"].as<std::string>();
          partitionPlanInfo_.partitionNames_ = splitString(input, delimiter);

          int idx = 0;
          for(size_t idx = 0; idx < partitionPlanInfo_.partitionNames_.size(); idx++) {
            partitionPlanInfo_.partitionBackendMap_.insert({partitionPlanInfo_.partitionNames_[idx], bnames[idx]});
            partitionPlanInfo_.partitionEmitMap_.insert({partitionPlanInfo_.partitionNames_[idx], enames[idx]});
          }
        }

        if (node["nodeToPartition"]) {
//          std::cout << "nodeToPartition: "
//                    << (node["nodeToPartition"].as<std::string>())
//                    << std::endl;
          input = node["nodeToPartition"].as<std::string>();
          nplist = splitString(input, delimiter);

          for(auto npstr: nplist) {
            if(npstr.empty())
              continue;

//            std::cout << "node-partition = " << npstr << std::endl;
            std::vector<std::string> np = splitString(npstr, '-');
            rtrim(np[0]);
            rtrim(np[1]);
            partitionPlanInfo_.nodeNames_.push_back(np[0]);
            partitionPlanInfo_.nodePartitionMap_.insert({np[0], np[1]});
          }
        }
      }
    }
  }catch (const YAML::Exception &e) {
    std::cerr << "Error loading the partition-plan YAML file: " << e.what() << std::endl;
  }
}

void Partition::loadDeviceConfig(std::string deviceConfig) {
  if (deviceConfig.empty()) {
    std::cout << "No deviceConfig file: " << std::endl;
    return;
  }

  try {
    YAML::Node deviceConfigInfo = YAML::LoadFile(deviceConfig);
    if (deviceConfigInfo.IsSequence()) {
      for (size_t i = 0; i < deviceConfigInfo.size(); ++i) {
        const YAML::Node &node = deviceConfigInfo[i];

        //        std::cout << "Item " << i + 1 << std::endl;
        std::cout << "Name: " << node["name"].as<std::string>() << std::endl;
        //        std::cout << "Backend Name: " << node["backendName"].as<std::string>() << std::endl; std::cout << "Device ID: " << node["deviceID"].as<int>() << std::endl;

        if (node["isDefaultDevice"]) {
          std::cout << "Is Default Device: "
                    << (node["isDefaultDevice"].as<bool>() ? "true" : "false")
                    << std::endl;
        }

        //        std::cout << "Supported Ops: " << node["supportedOps"].as<std::string>() << std::endl; std::cout << "Emit Type: " << node["emitType"].as<std::string>() << std::endl; std::cout << "----------------------" << std::endl;

        std::string deviceName = node["name"].as<std::string>();

        if (deviceConfigMap_->find(deviceName) == deviceConfigMap_->end()) {
          // itemMap = std::make_unique<std::map<std::string, std::string>>();
          (*deviceConfigMap_)[deviceName] =
              std::make_unique<std::map<std::string, std::string>>();
        }
        std::map<std::string, std::string>* itemMap =
            ((*deviceConfigMap_)[deviceName]).get();

        (*itemMap)["backendName"] = node["backendName"].as<std::string>();
        (*itemMap)["deviceID"] = std::to_string(node["deviceID"].as<int>());
        (*itemMap)["supportedOps"] = node["supportedOps"].as<std::string>();
        (*itemMap)["emitType"] = node["emitType"].as<std::string>();

        if (node["isDefaultDevice"]) {
          (*itemMap)["isDefaultDevice"] =
              (node["isDefaultDevice"].as<bool>() ? "true" : "false");
        } else {
          (*itemMap)["isDefaultDevice"] = "false";
        }
      }
    }
  } catch (const YAML::Exception &e) {
    std::cerr << "Error loading the device-config YAML file: " << e.what() << std::endl;
  }

  //  std::cout << "(CompilerUtils) deviceConfigMap size = " << deviceConfigMap.size() << std::endl;
}

void replaceWord(std::string& source, const std::string& oldWord, const std::string& newWord) {
  size_t pos = 0;
  if ((pos = source.find(oldWord, 0)) != std::string::npos) {
    source.replace(pos, oldWord.length(), newWord);
  }
}


void Partition::genUserMainFile(std::vector<std::string>& funcNameList, std::vector<std::string>& outputNameList) {
  srand(time(0));
//  std::cout << "(genUserMainFile) inOutDims_ = " << inOutDims_.at(0).first << " - " << inOutDims_.at(0).second << std::endl;

    //read main.cpp, input.h, CMakeLists.txt from /home/msyu/CLionProjects/onnx-mlir-pt/src/templates/
    std::cout << "== getUserMainFile == " << std::endl;

    std::string rootDir;// = getenv("ONNX_MLIR_ROOT");
    std::string runtimeDir;// = getenv("ONNX_MLIR_RUNTIME_DIR");

    std::cout << "rootDir = " << rootDir << std::endl;
    std::cout << "runtimeDir = " << runtimeDir << std::endl;

    std::cout << "emitFolder = " << emitFolder_ << std::endl;
    if(emitFolder_.empty()) {
      std::cerr << "no emit-folder name" << std::endl;
      return;
    }

    if(rootDir.empty()) {
      rootDir = "/home/msyu/CLionProjects/onnx-mlir-part";
      runtimeDir = "/home/msyu/CLionProjects/onnx-mlir-part/cmake-build-debug/Debug/lib";
    }

//    if(!rootDir.empty()) {

      std::string templateDir = "";
//      std::string outDir = rootDir + "/cmake-build-debug/Debug/bin/";
      templateDir = templateDir + rootDir + "/src/templates/";

      if (!llvm::sys::fs::is_directory(emitFolder_)) {
        llvm::sys::fs::create_directory(emitFolder_);
      }

      for(auto dim: inOutDims_) {
        //        std::cout << "dynEntryPointName = " << dynEntryPointName << std::endl;
        //    std::cout << "inOutDims size = " << inOutDims_.size()<< std::endl;
        std::cout << "dim " << dim.first << ", " << dim.second << std::endl;
        //        std::cout << "inOutDims[1] " << inOutDims_.at(1).first << ", "
        //                  << inOutDims_.at(1).second << std::endl;
      }
      std::ifstream mainFile(templateDir + "main.cpp");
      std::ifstream makeFile(templateDir + "CMakeLists.txt");

      llvm::sys::fs::copy_file(templateDir + "input.h", emitFolder_ + "/input.h");
      llvm::sys::fs::copy_file(templateDir + "test.png", emitFolder_ + "/test.png");

      std::string cmakeBinObjListStr = "";
      size_t idx = 0;
      for(auto objFile: outputNameList) {
        cmakeBinObjListStr = cmakeBinObjListStr + "\t\t${CMAKE_CURRENT_BINARY_DIR}/" + objFile + ".o\n";
        llvm::sys::fs::rename(
            objFile + ".o", emitFolder_ + "/" + objFile + ".o");
      }


      if (mainFile.is_open()) {

        std::string funcDefStr = "";
        std::string funcCallStr = "";
        idx = 0;
        for(auto pFuncName: funcNameList) {
          funcDefStr = funcDefStr + "extern \"C\" OMTensorList *" + pFuncName + "(OMTensorList *);\n";

          std::string tensorVarName = std::string("tensor_p") + std::to_string(idx);
          if(idx == 0) {
            funcCallStr = "OMTensor *" + tensorVarName + " = omTensorCreateWithOwnership(img_data, shape, rank, ONNX_TYPE_FLOAT, true);\n";
          } else {
            funcCallStr = funcCallStr + "OMTensor *" + tensorVarName + " = omTensorListGetOmtByIndex(tensorListOut, 0);\n";
          }
          //  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
          funcCallStr = funcCallStr + "inputTensors[0] = " + tensorVarName + ";\n";
          funcCallStr = funcCallStr + "OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);\n";
          funcCallStr = funcCallStr + "OMTensorList *tensorListOut = " + pFuncName + "(tensorListIn);\n\n";

          idx++;
        }

        std::ofstream newMainFile( emitFolder_+ "/main.cpp");
        std::string line;
        while (std::getline(mainFile, line)) {
          //std::cout << line << std::endl;
          replaceWord(line, "#FUNC_DEF#", funcDefStr);
          replaceWord(line, std::string("#RANK#"), std::to_string(inOutDims_.at(0).first));
          replaceWord(line, "#DIM_SHAPE#", inOutDims_.at(0).second);
          replaceWord(line, "#TENSOR_OP#", funcCallStr);
          newMainFile << line << std::endl;
        }

        newMainFile.close();
        mainFile.close();
      } else {
        std::cerr << "no main.cpp" << std::endl;
        return;
      }

      if (makeFile.is_open()) {
        std::ofstream newMakeFile(emitFolder_ + "/CMakeLists.txt");
        std::string line;
        int randomNumber = 1000 + rand() % 9000;
        while (std::getline(makeFile, line)) {
//          std::cout << line << std::endl;
          replaceWord(line, "#BIN_OBJ_NAMES#", cmakeBinObjListStr);
          replaceWord(line, "#EXE_NAME#", "partMlirExe" + std::to_string(randomNumber));
//          replaceWord(line, "#ONNX_MLIR_RUNTIME_DIR#", runtimeDir);
          replaceWord(line, "#MLIR_ROOT#", rootDir);
          newMakeFile << line << std::endl;
        }
        newMakeFile.close();
        makeFile.close();
      } else {
        std::cerr << "no CMakeLists.txt" << std::endl;
        return;
      }

//    } else {
//      return;
//    }
}
//
//void ConvertKrnlToLLVMPass::genUserMainFile(ModuleOp &module) {
//
//  //read main.cpp, input.h, CMakeLists.txt from /home/msyu/CLionProjects/onnx-mlir-pt/src/templates/
//  std::cout << "== getUserMainFile == " << std::endl;
//  //  const char* rootDir = getenv("ONNX_MLIR_ROOT");
//  //  std::string runtimeDir = getenv("ONNX_MLIR_RUNTIME_DIR");
//  const char* rootDir = "/home/msyu/CLionProjects/onnx-mlir-pt";
//  std::string runtimeDir = "/home/msyu/CLionProjects/onnx-mlir-pt/cmake-build-release/Release/lib";
//
//  std::cout << "rootDir = " << rootDir << std::endl;
//  std::cout << runtimeDir << std::endl;
//  std::cout << "emitFolder = " << emitFolder_ << std::endl;
//
//  if(emitFolder_.empty()) {
//    std::cerr << "no emit-folder name" << std::endl;
//    return;
//  }
//  //
//  //    std::cout << "ONNX_MLIR_ROOT = " << rootDir << std::endl;
//
//  if(rootDir != nullptr) {
//    std::string templateDir = "";
//    std::string outDir = emitFolder_ + "/";
//    templateDir = templateDir + rootDir + "/src/templates/";
//
//    if (!llvm::sys::fs::is_directory(emitFolder_)) {
//      llvm::sys::fs::create_directory(emitFolder_);
//    }
//
//    std::cout << "dynEntryPointName = " << dynEntryPointName << std::endl;
//    //    std::cout << "inOutDims size = " << inOutDims_.size()<< std::endl;
//    std::cout << "inOutDims[0] " << inOutDims_.at(0).first << ", " << inOutDims_.at(0).second << std::endl;
//    std::cout << "inOutDims[1] " << inOutDims_.at(1).first << ", " << inOutDims_.at(1).second << std::endl;
//
//    std::ifstream mainFile(templateDir + "main.cpp");
//    std::ifstream makeFile(templateDir + "CMakeLists.txt");
//
//    llvm::sys::fs::copy_file(templateDir + "input.h", outDir + "input.h");
//    llvm::sys::fs::copy_file(templateDir + "test.png", outDir + "test.png");
//    llvm::sys::fs::copy_file(outputNameNoExt + ".o", outDir + outputNameNoExt + ".o");
//
//    if (mainFile.is_open()) {
//
//      std::ofstream newMainFile(outDir  + "main.cpp");
//
//      std::string line;
//      while (std::getline(mainFile, line)) {
//        //std::cout << line << std::endl;
//        replaceWord(line, std::string("#RANK#"), std::to_string(inOutDims_.at(0).first));
//        replaceWord(line, "#DIM_SHAPE#", inOutDims_.at(0).second);
//        replaceWord(line, "#ENTRY_NAME#", dynEntryPointName);
//        newMainFile << line << std::endl;
//      }
//
//      newMainFile.close();
//      mainFile.close();
//    } else {
//      std::cerr << "no main.cpp" << std::endl;
//      return;
//    }
//
//    if (makeFile.is_open()) {
//
//      std::ofstream newMakeFile(outDir + "CMakeLists.txt");
//      std::string line;
//      while (std::getline(makeFile, line)) {
//        //std::cout << line << std::endl;
//        replaceWord(line, "#OBJ_NAME#", outputNameNoExt);
//        replaceWord(line, "#EXE_NAME#", outputNameNoExt + "Exe");
//        replaceWord(line, "#ONNX_MLIR_RUNTIME_DIR#", runtimeDir);
//        replaceWord(line, "#ONNX_MLIR_ROOT#", rootDir);
//
//        newMakeFile << line << std::endl;
//      }
//
//      newMakeFile.close();
//      makeFile.close();
//    } else {
//      std::cerr << "no CMakeLists.txt" << std::endl;
//      return;
//    }
//
//  } else {
//    return;
//  }
//}


}