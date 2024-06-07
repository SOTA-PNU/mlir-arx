
#include <chrono>
#include <sstream>
#include "OnnxMlirRuntime.h"

//
//
//extern "C" OMTensorList *run_p0_robertabaseo3(OMTensorList *);
//extern "C" OMTensorList *run_p1_robertabaseo3(OMTensorList *);
//
//
//int main(int argc, char **argv) {
//
//  // Create an input tensor list of 1 tensor.
//  int inputNum = 1;
//  OMTensor *inputTensors[inputNum];
//  // Initialize input data (Assuming the use case for a BERT model)
//  int64_t input_ids[] = {101, 7592, 1010, 2129, 2024, 2017, 102}; //We must use hugging face tokenize library.
//
//  int64_t rank = 2; //2D
//  int64_t input_shape[] = {1, static_cast<int64_t>(7)};
//
//  OMTensor *tensor = omTensorCreateWithOwnership(input_ids, input_shape, rank, ONNX_TYPE_INT64, /*owning=*/true);
//
//  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
//  inputTensors[0] = tensor;
//  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);
//
//  auto start = std::chrono::high_resolution_clock::now();
//  // Compute outputs.
//  OMTensorList *tensorListOut = run_p0_robertabaseo3(tensorListIn);
//
//  OMTensor *tensor_p1 = omTensorListGetOmtByIndex(tensorListOut, 0);
//  inputTensors[0] = tensor_p1;
//  tensorListIn = omTensorListCreate(inputTensors, inputNum);
//  tensorListOut = run_p1_robertabaseo3(tensorListIn);
//
//  auto stop = std::chrono::high_resolution_clock::now();
//
//  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//  std::cout << "Inference time: "
//            << duration.count() << " msec" << std::endl;
//
//  if (tensorListOut == NULL) {
//    std::cerr << "Model execution failed, tensorListOut is NULL." << std::endl;
//    omTensorListDestroy(tensorListIn);  // Cleanup input tensors
//    return -1;  // Exit with an error code
//  }
//
//  OMTensor *outputTensor = omTensorListGetOmtByIndex(tensorListOut, 0);
//  float *outputData = (float *)omTensorGetDataPtr(outputTensor);
//  std::cout << "Model output:" << std::endl;
//  for (int i = 0; i < omTensorGetNumElems(outputTensor); i++) {
//    std::cout << outputData[i] << " ";
//  }
//  std::cout << std::endl;
//
//
//
//
//
//  //  // The OMTensorListDestroy will free all tensors in the OMTensorList
//  //  // upon destruction. It is important to note, that every tensor will
//  //  // be destroyed. To free the OMTensorList data structure but leave the
//  //  // tensors as is, use OMTensorListDestroyShallow instead.
//
//  omTensorListDestroy(tensorListOut);
//  //  omTensorListDestroy(tensorListIn);
//
//  //  printf("idx = %d\n", digit);
//  return 0;
//}
//
extern "C" OMTensorList *run_main_graph(OMTensorList *);
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
  int inputNum = 1;
  OMTensor *inputTensors[inputNum];
  // Initialize input data (Assuming the use case for a BERT model)
  int64_t input_ids[] = {101, 7592, 1010, 2129, 2024, 2017, 102}; //We must use hugging face tokenize library.

  int64_t rank = 2; //2D
  int64_t input_shape[] = {1, static_cast<int64_t>(7)};

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
