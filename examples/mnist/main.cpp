//#include <iostream>
#include <vector>
//#include <assert.h>
//#include <png.h>

#include "input.h"
#include "OnnxMlirRuntime.h"

// Declare the inference entry point.
extern "C" OMTensorList *run_main_graph(OMTensorList *);

int main(int argc, char **argv) {

  // Create an input tensor list of 1 tensor.
  int inputNum = 1;
  OMTensor *inputTensors[inputNum];
  // The first input is of tensor<1x1x28x28xf32>.
  int64_t rank = 4;
  int64_t shape[] = {1, 1, 28, 28};


//  size_t resultSizeInBytes = 28*28*sizeof(float);
//  printf("resultSizeInBytes = %lu\n", resultSizeInBytes);

//  static float* img_data_mnist = static_cast<float *>(malloc(resultSizeInBytes));;
//  static float img_data_mnist[10000];
  static float img_data[28*28];
  loadImagesAndPreprocess(argv[1], img_data, shape);
//  for(int i = 0; i < 10; i++) {
//    printf("data: %f\n", imgData[i]);
//  }


  // Create a tensor using omTensorCreateWithOwnership (returns a pointer to the OMTensor).
  // When the parameter, owning is set to "true", the OMTensor will free the data
  // pointer (img_data) upon destruction. If owning is set to false, the data pointer will
  // not be freed upon destruction.
  OMTensor *tensor = omTensorCreateWithOwnership(img_data, shape, rank, ONNX_TYPE_FLOAT, /*owning=*/true);

  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

  clock_t start, end;
  double result;
  start = clock();

  // Compute outputs.
  OMTensorList *tensorListOut = run_main_graph(tensorListIn);


  end = clock();
  result = (double)(end-start)/CLOCKS_PER_SEC*1000;
  std::cout<<"Inference time: "<<result<<"ms\n";


  // Extract the output. The model defines one output of type tensor<1x10xf32>.
  OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *prediction = (float *)omTensorGetDataPtr(y);

  // Analyze the output.
  int digit = -1;
  float prob = 0.;
  for (int i = 0; i < 10; i++) {
    printf("prediction[%d] = %f\n", i, prediction[i]);
    if (prediction[i] > prob) {
      digit = i;
      prob = prediction[i];
    }
  }
  // The OMTensorListDestroy will free all tensors in the OMTensorList
  // upon destruction. It is important to note, that every tensor will
  // be destroyed. To free the OMTensorList data structure but leave the
  // tensors as is, use OMTensorListDestroyShallow instead.

  omTensorListDestroy(tensorListOut);
//  omTensorListDestroy(tensorListIn);

  printf("The digit is %d\n", digit);
  return 0;
}
