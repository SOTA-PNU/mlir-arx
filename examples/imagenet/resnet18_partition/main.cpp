//#include <iostream>
#include <vector>
//#include <assert.h>
//#include <png.h>

#include "input.h"
#include "OnnxMlirRuntime.h"
//
//// Declare the inference entry point.
extern "C" OMTensorList *run_p0_resnet18o3(OMTensorList *);
extern "C" OMTensorList *run_p1_resnet18o3(OMTensorList *);

int main(int argc, char **argv) {

  // Create an input tensor list of 1 tensor.
  OMTensor *inputTensors[2];
  // The first input is of tensor<1x1x28x28xf32>.
  int64_t rank = 4;
  int64_t shape[] = {1 , 224 , 224, 3 };

  static float img_data[224*224*3];
//  loadImagesAndPreprocess(argv[1], img_data, shape);
  loadImagesAndPreprocess(argv[1], img_data);

  // Create a tensor using omTensorCreateWithOwnership (returns a pointer to the OMTensor).
  // When the parameter, owning is set to "true", the OMTensor will free the data
  // pointer (img_data) upon destruction. If owning is set to false, the data pointer will
  // not be freed upon destruction.
  OMTensor *tensor = omTensorCreateWithOwnership(img_data, shape, rank, ONNX_TYPE_FLOAT, /*owning=*/true);

//  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, 1);
//  // Compute outputs.
  OMTensorList *tensorListOut = run_p0_resnet18o3(tensorListIn);
////
////  // Extract the output. The model defines one output of type tensor<1x10xf32>.
  OMTensor *tensor_p0_out0 = omTensorListGetOmtByIndex(tensorListOut, 0);
  OMTensor *tensor_p0_out1 = omTensorListGetOmtByIndex(tensorListOut, 1);

  inputTensors[0] = tensor_p0_out1;
  inputTensors[1] = tensor_p0_out0;

  tensorListIn = omTensorListCreate(inputTensors, 2);
  tensorListOut = run_p1_resnet18o3(tensorListIn);

  OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *prediction = (float *)omTensorGetDataPtr(y);

  // Analyze the output.
  int digit = -1;
  float prob = 0.;
  for (int i = 0; i < 1000; i++) {
//    printf("prediction[%d] = %f\n", i, prediction[i]);
    if (prediction[i] > prob) {
      digit = i;
      prob = prediction[i];
    }
  }

  printf("idx = %d\n", digit);
//
//  // The OMTensorListDestroy will free all tensors in the OMTensorList
//  // upon destruction. It is important to note, that every tensor will
//  // be destroyed. To free the OMTensorList data structure but leave the
//  // tensors as is, use OMTensorListDestroyShallow instead.
//
  omTensorListDestroy(tensorListOut);
  omTensorListDestroy(tensorListIn);

  assert(digit == 285);

//  printf("Te end of main()\n");
  return 0;
}
