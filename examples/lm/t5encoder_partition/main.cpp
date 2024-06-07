
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
//  std::string sentence = "Hello world, this is an example sentence.";
//  auto tokens = tokenize(sentence);
//
//  // 예제를 위한 토큰 ID 할당 (실제 사용 시 사전에 매핑된 ID 사용 필요)
//  std::vector<float> token_ids;
//  for (const auto &token : tokens) {
//    token_ids.push_back(static_cast<float>(token.size()));  // 예제에서는 토큰 길이를 ID로 사용
//  }

  // Create an input tensor list of 1 tensor.
//  int inputNum = 1;
//  OMTensor *inputTensors[inputNum];
  // Initialize input data (Assuming the use case for a BERT model)
  int64_t input_ids[10] = {0, 2003, 1996, 3582, 2829, 102, 0, 0, 0, 0};
  int64_t attention_mask[10] = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0};

  int64_t rank = 2; //2D
  int64_t input_shape[] = {1, static_cast<int64_t>(10)};

  OMTensor *tensor_in = omTensorCreateWithOwnership(input_ids, input_shape, rank, ONNX_TYPE_INT64, /*owning=*/true);
  OMTensor *tensor_mask = omTensorCreateWithOwnership(attention_mask, input_shape, rank, ONNX_TYPE_INT64, /*owning=*/true);

  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
  OMTensor* inputTensors[2] = {tensor_in, tensor_mask};
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, 2);

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
  for (int i = 0; i < omTensorGetNumElems(outputTensor); i++) {
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
