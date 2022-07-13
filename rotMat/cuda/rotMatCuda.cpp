// Largely boilerplate gateway to cuda-enabled routines

#include <torch/extension.h>

#include <vector>

// Declarations to cuda-enabled functions
torch::Tensor rotMatForwardCuda( torch::Tensor thetas, int64_t N, int threadsPerBlock);
torch::Tensor rotMatBackwardCuda( torch::Tensor thetas, torch::Tensor U, torch::Tensor G, int threadsPerBlock );

// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor rotMatForward( torch::Tensor thetas, int64_t N, int threadsPerBlock)
{

  CHECK_INPUT(thetas);

  return rotMatForwardCuda( thetas, N, threadsPerBlock );
}


torch::Tensor rotMatBackward( torch::Tensor thetas, torch::Tensor U, torch::Tensor G, int threadsPerBlock )
{

  CHECK_INPUT(thetas);
  CHECK_INPUT(U);
  CHECK_INPUT(G);

  return rotMatBackwardCuda( thetas, U, G, threadsPerBlock);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rotMatForward, "Rotation Matrix forward (cuda)");
  m.def("backward", &rotMatBackward, "Rotation Matrix backward (cuda)");
}
