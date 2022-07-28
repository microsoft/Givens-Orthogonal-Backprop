// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <vector>

// Declarations to cuda-enabled functions
torch::Tensor rotMatForwardCuda( torch::Tensor X, torch::Tensor thetas);
torch::Tensor rotMatBackwardCuda(torch::Tensor thetas, torch::Tensor U, torch::Tensor G);

// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor rotMatForward(torch::Tensor X, torch::Tensor thetas)
{
  CHECK_INPUT(thetas);
  CHECK_INPUT(X);
  return rotMatForwardCuda(X, thetas);
}

torch::Tensor rotMatBackward( torch::Tensor thetas, torch::Tensor U, torch::Tensor G)
{
  CHECK_INPUT(thetas);
  CHECK_INPUT(U);
  CHECK_INPUT(G);
  return rotMatBackwardCuda(thetas, U, G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rotMatForward, "Rotation Matrix forward (cuda)");
  m.def("backward", &rotMatBackward, "Rotation Matrix backward (cuda)");
}
