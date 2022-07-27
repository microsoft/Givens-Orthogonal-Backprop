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
  __global__ void updateGivensElements(
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> U,
    const int deadIndex,
    const int Ntilde,
    const int dMax,
    const int tournamentStep)
{
  // If transpose k works on rows; otherwise on columns
  const int k = threadIdx.y + blockDim.y*blockIdx.y;
  
  const int N = U.size(0);
  const int M = U.size(1);
  if (k >= M)
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

  const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
  const scalar_t cij = C[thetaIndex];
  const scalar_t sij = S[thetaIndex];

  // Apply Givens: Update U's offsets
  const scalar_t Ui = U[i][k];
  const scalar_t Uj = U[j][k];

  U[i][k] = Ui*cij - Uj*sij;
  U[j][k] = Ui*sij + Uj*cij;
}

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

template <typename scalar_t> 
  __global__ void setJVP(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> M,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UfTrans,
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
  
  const int N = UfTrans.size(0);
  if (k >= UfTrans.size(1))
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
  const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
  const scalar_t cij = C[thetaIndex];
  const scalar_t sij = -1 * S[thetaIndex];

  // Apply Givens: Update U's offsets
  const scalar_t Ui = UfTrans[i][k];
  const scalar_t Uj = UfTrans[j][k];

  const scalar_t newUfTik = Ui*cij - Uj*sij;
  UfTrans[i][k] = newUfTik;

  const scalar_t newUfTjk = Ui*sij + Uj*cij;
  UfTrans[j][k] = newUfTjk;

  // Repeat for M
  const scalar_t Mi = M[i][k];
  const scalar_t Mj = M[j][k];

  const scalar_t newMik = Mi*cij - Mj*sij;
  M[i][k] = newMik;

  const scalar_t newMjk = Mi*sij + Mj*cij;
  M[j][k] = newMjk;

  // Set A, skip a write if possible can
  sA[tid] = newMik * newUfTjk - newMjk * newUfTik;
  /*__syncthreads();

  // Reduce
  if (ThreadsPerRowBackward == 1024) {
    if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads();}
  if (ThreadsPerRowBackward >= 512) {
    if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads();}
  if (ThreadsPerRowBackward >= 256) {
    if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
  if (ThreadsPerRowBackward >= 128) {
    if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
  if(tid <32) warpReduceAtBackward(sA,tid);*/
  
  //if (tid == 0) 
  atomicAdd(&JVP[thetaIndex], sA[tid]);
}

std::tuple<int, int, int> determineRotMatConstants(const int nThetas, const int N)
{
  // if nThetas == maxPairs
  auto dMax = N-1;
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

torch::Tensor rotMatForwardCuda(torch::Tensor Uorg, torch::Tensor thetas, int N)
{
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto deadIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  // Set U same device and type as thetas
  auto tensOptions = torch::TensorOptions()
    .dtype(thetas.dtype())
    .device(thetas.device());

  auto U = torch::clone(Uorg).detach();
  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  // CUDA grid: blocks of size: Ntilde/2 x ceil(N/nThreads)
  const int nBlocksY = U.size(1)/ThreadsPerRowForward + (U.size(1)% ThreadsPerRowForward != 0);
  const int nBlocksX = Ntilde / 2;

  const dim3 blocks(nBlocksX, nBlocksY);
  const dim3 threads(1, ThreadsPerRowForward);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. There are n-1 steps
  for (int tournamentStep=Ntilde-2; tournamentStep>=0; tournamentStep--)
  {
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatForwardCuda",
      ([&]{
        updateGivensElements<scalar_t><<<blocks,threads>>>(
          C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          U.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          deadIndex,Ntilde, dMax, tournamentStep);
      }));
  }

  return U;
}

torch::Tensor rotMatBackwardCuda(
  torch::Tensor X,
  torch::Tensor thetas,
  torch::Tensor U,
  torch::Tensor G)
{
  auto N = U.size(0);
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto deadIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  // In rotMatForwardCuda, U is given these same properties
  auto tensOptions = torch::TensorOptions()
    .dtype(thetas.dtype())
    .device(thetas.device());

  //auto M = torch::eye(N,N, tensOptions);
  //M.index_put_({Slice(0, G.size(1)), Slice()}, G.detach().t());
  auto M = G.t().contiguous().detach();
  M = torch::matmul(U, M);

  auto UfTrans = torch::eye(N,N, tensOptions);
  //UfTrans.index_put_({Slice(0, U.size(1)), Slice()}, U.detach().t());
  
  ///auto UfTrans = torch::clone(U);// .t().contiguous().detach();

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());
  auto JVP = torch::zeros_like(thetas, tensOptions);

  // CUDA grid: blocks of size: Ntilde/2 x ceil(N/nThreads)
  const int nBlocksX = Ntilde / 2;
  const int nBlocksY = N/ThreadsPerRowBackward + (N % ThreadsPerRowBackward != 0);

  const dim3 blocks(nBlocksX, nBlocksY);
  const dim3 threads(1, ThreadsPerRowBackward);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. here are n-1 steps
  for (int tournamentStep=0; tournamentStep<=Ntilde-2; tournamentStep++)
  {
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatBackwardCuda",
      ([&]{
        setJVP<scalar_t><<<blocks,threads>>>(
          M.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          UfTrans.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
          C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          JVP.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
          deadIndex, Ntilde, dMax, tournamentStep);
      }));
  }
  
  return JVP;
}
