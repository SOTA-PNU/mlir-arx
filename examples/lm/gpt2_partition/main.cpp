
#include <chrono>
#include <sstream>
#include "OnnxMlirRuntime.h"

// Declare the inference entry point.
extern "C" OMTensorList *run_main_graph(OMTensorList *);

std::vector<std::string> tokenize(const std::string &sentence) {
std::istringstream iss(sentence);
std::vector<std::string> tokens;
std::string token;
while (iss >> token) {
  tokens.push_back(token);
}
return tokens;
}

int main(int argc, char **argv) {
  // Create an input tensor list of 1 tensor.
  int inputNum = 1;
  OMTensor *inputTensors[inputNum];
  // Initialize input data (Assuming the use case for a BERT model)
  int64_t input_ids[1][9] = {4342, 318, 617, 2420, 284, 37773, 1058, 18435, 2159};

  int64_t rank = 3; //2D
  int64_t input_shape[] = {1, static_cast<int64_t>(9)};

  OMTensor *tensor = omTensorCreateWithOwnership(input_ids, input_shape, rank, ONNX_TYPE_INT64, /*owning=*/true);

  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

  auto start = std::chrono::high_resolution_clock::now();
  // Compute outputs.
  OMTensorList *tensorListOut = run_main_graph(tensorListIn);
  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Inference time: "
            << duration.count() << " msec" << std::endl;

  if (tensorListOut == NULL) {
    std::cerr << "Model execution failed, tensorListOut is NULL." << std::endl;
    omTensorListDestroy(tensorListIn);  // Cleanup input tensors
    return -1;  // Exit with an error code
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *outputData = (float *)omTensorGetDataPtr(outputTensor);
  std::cout << "Model output:" << std::endl;
//  for (int i = 0; i < omTensorGetNumElems(outputTensor); i++) {
  for (int i = 0; i < 1; i++) {
    std::cout << outputData[i] << " ";
  }
  std::cout << std::endl;


//  // The OMTensorListDestroy will free all tensors in the OMTensorList
//  // upon destruction. It is important to note, that every tensor will
//  // be destroyed. To free the OMTensorList data structure but leave the
//  // tensors as is, use OMTensorListDestroyShallow instead.

  omTensorListDestroy(tensorListOut);
  //  omTensorListDestroy(tensorListIn);

//  printf("idx = %d\n", digit);
  return 0;
}
