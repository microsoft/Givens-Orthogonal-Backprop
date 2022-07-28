// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include <functional>
#include <vector>

using namespace torch::indexing;

#define WarpSize 32
#define ThreadsPerRowForward 128
#define ThreadsPerRowBackward 256

// https://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles
// https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(
  double* address, 
  double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename scalar_t>
  __device__ void warpReduceAtBackward(
    volatile scalar_t* sdata, 
    int tid)
{
  if (ThreadsPerRowBackward >= 64)  sdata[tid] += sdata[tid + 32];
  if (ThreadsPerRowBackward >= 32) sdata[tid] += sdata[tid + 16];
  if (ThreadsPerRowBackward >= 16) sdata[tid] += sdata[tid + 8];
  if (ThreadsPerRowBackward >= 8) sdata[tid] += sdata[tid + 4];
  if (ThreadsPerRowBackward >= 4) sdata[tid] += sdata[tid + 2];
  if (ThreadsPerRowBackward >= 2) sdata[tid] += sdata[tid + 1];
}

__device__ __forceinline__ std::pair<const int, const int> determineRowIndexPair(
  const int blockIndex,
  const int Ntilde,
  const int tournamentStep)
{
  const int n = Ntilde - 1;

  // Determine i,j from block index
  int i = blockIndex==0 ? 0 : blockIndex  + tournamentStep;
  if (i >= Ntilde)
  {
    i -= n;
  }

  int j = n - blockIndex + tournamentStep;
  if (j >= Ntilde)
  {
    j -= n;
  }

  return i > j ? std::make_pair(j,i) : std::make_pair(i,j);
}

__device__ __forceinline__ bool areRowIndicesOutOfRange(
  const int i, 
  const int j, 
  const int deadIndex, 
  const int dMax)
{
  // check if the coordinates are out of range or equal dummy coordinate (dummy exists when N is odd)
  return j == deadIndex || (i > dMax && j > dMax);
}

 template <typename scalar_t>  
  __global__ void ApplyRoundRobinGivensRotationMatrix(
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> U,
    const int deadIndex,
    const int Ntilde,
    const int dMax,
    const int tournamentStep)
{
  const int k = threadIdx.y + blockDim.y*blockIdx.y;
  if (k >= U.size(1))
  {
    return;
  }

  auto rowIndices = determineRowIndexPair(blockIdx.x, Ntilde, tournamentStep);
  const int i = rowIndices.first;
  const int j = rowIndices.second;
  if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
  {
    return;
  }

  const int N = U.size(0);
  const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
  const scalar_t cij = C[thetaIndex];
  const scalar_t sij = S[thetaIndex];

  // Apply Givens
  const scalar_t Ui = U[i][k];
  const scalar_t Uj = U[j][k];

  U[i][k] = Ui*cij - Uj*sij;
  U[j][k] = Ui*sij + Uj*cij;
}

template <typename scalar_t> 
  __global__ void CalculateRoundRobinGivensThetaJVPs(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UX,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> G,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> JVP,
    const int deadIndex,
    const int Ntilde,
    const int dMax,
    const int tournamentStep)
{
  __shared__ scalar_t sA[ThreadsPerRowBackward];

  // k is the column index of M and the row index of Uf, to set col of A
  const int tid = threadIdx.y;
  const int k = tid + blockDim.y*blockIdx.y;
  if (k >= UX.size(1))
  {
    sA[tid] = 0;
    return;
  }

  auto rowIndices = determineRowIndexPair(blockIdx.x, Ntilde, tournamentStep);
  int i = rowIndices.first;
  int j = rowIndices.second;
  if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
  {
    sA[tid] = 0;
    return;
  }

   __syncthreads();
  const int N = UX.size(0);
  const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
  const scalar_t cij = C[thetaIndex];
  const scalar_t sij = -1 * S[thetaIndex]; // Transpose of a Givens rotation has the signs of sij flipped

  // Apply Givens Transpose to G
  const scalar_t Gi = G[i][k];
  const scalar_t Gj = G[j][k];

  const scalar_t newGi = Gi*cij - Gj*sij;
  G[i][k] = newGi;

  const scalar_t newGj = Gi*sij + Gj*cij;
  G[j][k] = newGj;

  // Repeat for UX
  const scalar_t UXi = UX[i][k];
  const scalar_t UXj = UX[j][k];

  const scalar_t newUXi = UXi*cij - UXj*sij;
  UX[i][k] = newUXi;

  const scalar_t newUXj = UXi*sij + UXj*cij;
  UX[j][k] = newUXj;

  sA[tid] = newUXi * newGj - newUXj * newGi;
  __syncthreads();

  // Reduce
  if (ThreadsPerRowBackward == 1024) {
    if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads();}
  if (ThreadsPerRowBackward >= 512) {
    if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads();}
  if (ThreadsPerRowBackward >= 256) {
    if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
  if (ThreadsPerRowBackward >= 128) {
    if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
  if(tid < 32) warpReduceAtBackward(sA, tid);
  
  if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
}

std::tuple<int, int, int> determineRotMatConstants(const int nThetas, const int N)
{
  auto dMax = N-1; // If nThetas == maxPairs
  if (nThetas < N*(N-1)/2)
  {
    dMax -= 1 + int(sqrt(1 - 4*(2*nThetas - N*(N-1)))) / 2;
  }

  // Handle odd N; in that case Ntilde is the even augmented dimension
  int deadIndex = -1;
  auto Ntilde = N;
  if (N % 2 != 0)
  {
    Ntilde += 1;
    deadIndex = Ntilde-1;
  }

  return std::make_tuple(dMax, deadIndex, Ntilde);
}

torch::Tensor rotMatForwardCuda(torch::Tensor X, torch::Tensor thetas)
{
  const int N = X.size(0);
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto deadIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  const int nBlocksY = X.size(1)/ThreadsPerRowForward + (X.size(1)% ThreadsPerRowForward != 0);
  const dim3 blocks(Ntilde / 2, nBlocksY);
  const dim3 threads(1, ThreadsPerRowForward);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. There are n-1 steps
  for (int tournamentStep=Ntilde-2; tournamentStep>=0; tournamentStep--)
  {
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatForwardCuda",
      ([&]{
        ApplyRoundRobinGivensRotationMatrix<scalar_t><<<blocks,threads>>>(
          C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          X.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          deadIndex,Ntilde, dMax, tournamentStep);
      }));
  }

  return X;
}

std::pair<torch::Tensor, torch::Tensor> rotMatBackwardCuda(
  torch::Tensor thetas,
  torch::Tensor UX,
  torch::Tensor G)
{
  auto N = UX.size(0);
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto deadIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  auto thetasTensorOptions = torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());
  auto JVP = torch::zeros_like(thetas, thetasTensorOptions);

  const int nBlocksY = N/ThreadsPerRowBackward + (N % ThreadsPerRowBackward != 0);
  const dim3 blocks(Ntilde / 2, nBlocksY);
  const dim3 threads(1, ThreadsPerRowBackward);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. here are n-1 steps
  for (int tournamentStep=0; tournamentStep<=Ntilde-2; tournamentStep++)
  {
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatBackwardCuda",
      ([&]{
        CalculateRoundRobinGivensThetaJVPs<scalar_t><<<blocks,threads>>>(
          UX.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          G.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          JVP.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          deadIndex, Ntilde, dMax, tournamentStep);
      }));
  }
  
  return std::make_pair(G, JVP);
}
