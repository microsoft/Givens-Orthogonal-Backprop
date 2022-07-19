// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include <functional>
#include <vector>

#define threadsPerBlock 16

using namespace torch::indexing;

__device__ __forceinline__ std::pair<const int, const int> determineRowIndexPair(
  const int blockIndex,
  const size_t Ntilde,
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

__device__ __forceinline__ bool areRowIndicesOutOfRange(const int i, const int j, const int deadIndex, const int dMax)
{
  // check if the coordinates are out of range or equal dummy coordinate (dummy exists when N is odd)
  return j == deadIndex || (i > dMax && j > dMax);
}

 __global__ void updateGivensElements(
  const float* const __restrict__ C,
  const float* const __restrict__ S,
  float* __restrict__ U,
  const size_t N,
  const int deadIndex,
  const size_t Ntilde,
  const int dMax,
  const int tournamentStep)
{
  // If transpose k works on rows; otherwise on columns
  const int k = threadIdx.y + blockDim.y*blockIdx.y;
  if (k >= N)
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
  const float cij = C[thetaIndex];
  const float sij = S[thetaIndex];

  const int iOffset = i*N + k;
  const int jOffset = j*N + k;

  // Apply Givens: Update U's offsets
  const float Ui = U[iOffset];
  const float Uj = U[jOffset];

  U[iOffset] = Ui*cij - Uj*sij;
  U[jOffset] = Ui*sij + Uj*cij;
}

__device__ void warpReduce(volatile float* sdata, int tid)
{
  if (threadsPerBlock >= 64)  sdata[tid] += sdata[tid + 32];
  if (threadsPerBlock >= 32) sdata[tid] += sdata[tid + 16];
  if (threadsPerBlock >= 16) sdata[tid] += sdata[tid + 8];
  if (threadsPerBlock >= 8) sdata[tid] += sdata[tid + 4];
  if (threadsPerBlock >= 4) sdata[tid] += sdata[tid + 2];
  if (threadsPerBlock >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void setA(
  const float* const __restrict__ C,
  const float* const __restrict__ S,
  float* __restrict__ M,
  float* __restrict__ UfTrans,
  at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> A,
  const int N,
  const int deadIndex,
  const int Ntilde,
  const int dMax,
  const int tournamentStep)
{

  extern __shared__ float sA[];

  // k is the column index of M and the row index of Uf, to set col of A
  const int tid = threadIdx.y;
  const int k = tid + blockDim.y*blockIdx.y;
  if (k >= N)
  {
    sA[tid] = 0;
    return;
  }

  auto rowIndices = determineRowIndexPair(blockIdx.x, Ntilde, tournamentStep);
  int i = rowIndices.first;
  int j = rowIndices.second;

  if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
  {
    return;
  }

  const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
  const float cij = C[thetaIndex];
  const float sij = S[thetaIndex];

  // Apply Givens: Update U's offsets
  const int ik = i*N + k;
  const int jk = j*N + k;

  const float Ui = UfTrans[ik];
  const float Uj = UfTrans[jk];

  const float newUfTik = Ui*cij - Uj*sij;
  UfTrans[ik] = newUfTik;

  const float newUfTjk = Ui*sij + Uj*cij;
  UfTrans[jk] = newUfTjk;

  // Repeat for M
  const float Mi = M[ik];
  const float Mj = M[jk];

  const float newMik = Mi*cij - Mj*sij;
  M[ik] = newMik;

  const float newMjk = Mi*sij + Mj*cij;
  M[jk] = newMjk;

  // Set A, skip a write if possible can
  sA[tid] = newMik * newUfTjk - newMjk * newUfTik;
  __syncthreads();

  // Reduce
  if (threadsPerBlock == 1024) {
    if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads();}
  if (threadsPerBlock >= 512) {
    if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads();}
  if (threadsPerBlock >= 256) {
    if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
  if (threadsPerBlock >= 128) {
    if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
  if(tid <32) warpReduce(sA,tid);
  
  if (tid == 0) A[thetaIndex][blockIdx.y] = sA[0];
}

 __global__ void setJVP(
  const float* __restrict__ Asum,
  float* __restrict__ JVP,
  const size_t N,
  const int deadIndex,
  const size_t Ntilde,
  const int dMax,
  const int tournamentStep)
{
  auto rowIndices = determineRowIndexPair(blockIdx.x, Ntilde, tournamentStep);
  int i = rowIndices.first;
  int j = rowIndices.second;

  if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
  {
    return;
  }

  const int thetaIndx = i*N - (i + 2)*(i + 1)/2 + j;
  JVP[thetaIndx] = Asum[blockIdx.x];
}

std::tuple<int64_t, int64_t, int64_t> determineRotMatConstants(const size_t nThetas, const size_t N)
{
  auto maxPairs = N*(N-1)/2;

  // if nThetas == maxPairs
  auto dMax = N-1;
  if (nThetas < maxPairs)
  {
    auto K = int(1 + sqrt(1 - 4*(2*nThetas - N*(N-1)))) / 2;
    dMax -= K;
  }

  // Handle odd N; in that case Ntilde is the even augmented dimension
  int64_t deadIndex = -1;
  auto Ntilde = N;
  if (N % 2 != 0)
  {
    Ntilde += 1;
    deadIndex = Ntilde-1;
  }

  return std::make_tuple(dMax, deadIndex, Ntilde);
}

torch::Tensor rotMatForwardCuda(torch::Tensor thetas, int64_t N)
{
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto deadIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  // Set U same device and type as thetas
  auto tensOptions = torch::TensorOptions()
    .dtype(thetas.dtype())
    .device(thetas.device());

  auto U = torch::eye(N, tensOptions);
  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  // CUDA grid: blocks of size: Ntilde/2 x ceil(N/nThreads)
  const int nBlocksY = N/threadsPerBlock + (N % threadsPerBlock != 0);
  const int nBlocksX = Ntilde / 2;

  const dim3 blocks(nBlocksX, nBlocksY);
  const dim3 threads(1, threadsPerBlock);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. There are n-1 steps
  for (int64_t tournamentStep=Ntilde-2; tournamentStep>=0; tournamentStep--)
  {
      updateGivensElements<<<blocks,threads>>>(
        C.data_ptr<float>(),
        S.data_ptr<float>(),
        U.data_ptr<float>(),
        N, deadIndex, Ntilde, dMax, tournamentStep);
  }

  return U;
}

torch::Tensor rotMatBackwardCuda(
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

  // Loss gradient wrt U params
  auto JVP = torch::zeros_like(thetas, tensOptions);

  auto M = G.t().contiguous().detach();
  auto UfTrans = U.t().contiguous().detach();

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  // CUDA grid: blocks of size: Ntilde/2 x ceil(N/nThreads)
  const int nBlocksX = Ntilde / 2;
  const int nBlocksY = N/threadsPerBlock + (N % threadsPerBlock != 0);
  
  // Note 0 dim is Ntilde not N
  auto A = torch::zeros({thetas.size(0), nBlocksY}, tensOptions).detach();

  const dim3 blocks(nBlocksX, nBlocksY);
  const dim3 threads(1, threadsPerBlock);

  //cudaStream_t stream1;
  //cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel.
  // There are n-1 steps
  for (int tournamentStep=Ntilde-2; tournamentStep>=0; tournamentStep--)
  {
    // Set A els
    setA<<<blocks,threads, sizeof(float) * threadsPerBlock>>>(
        C.data_ptr<float>(),
        S.data_ptr<float>(),
        M.data_ptr<float>(),
        UfTrans.data_ptr<float>(),
        A.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
        N, deadIndex, Ntilde, dMax, tournamentStep);

    // Sum rows
    //cudaStreamSynchronize(stream1);
    //auto Asum = A.sum(1);

    // Parallelized setting of JVP els from Asum; note here Ntilde/2 blocks, 1 thr/blck
    //setJVP<<<nBlocksX,1,0,stream1>>>(Asum.data_ptr<float>(), JVP.data_ptr<float>(), N, deadIndex, Ntilde, dMax, tournamentStep);
  }
  
  //cudaStreamSynchronize(stream1);
  return A.sum(1);
}
