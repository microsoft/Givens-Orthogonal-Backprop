// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include <functional>
#include <vector>

#include <string>
#include <iostream>

using namespace std;

using namespace torch::indexing;

__device__ __forceinline__ std::pair<const int, const int> determineRowIndexPair(
  const int blockIndex,
  const size_t playerCount,
  const int tournamentStep)
{
  const int n = playerCount - 1;

  // Determine i,j from block index
  int i = blockIndex==0 ? 0 : blockIndex  + tournamentStep;
  if (i >= playerCount)
  {
    i -= n;
  }

  int j = n - blockIndex + tournamentStep;
  if (j >= playerCount)
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

template <typename scalar_t> __global__ void updateGivensElementsRoundRobin(
  const scalar_t* const __restrict__ C,
  const scalar_t* const __restrict__ S,
  scalar_t* __restrict__ U,
  const int N,
  const int deadIndex,
  const int dMax)
{
  // If transpose k works on rows; otherwise on columns
  const int col = threadIdx.x + blockDim.x*blockIdx.x; 
  if (col >= N)
  {
    return;
  }

  const int playerCountInBlock = (blockDim.y *2);
  const int blockStart = playerCountInBlock * blockIdx.y;

  //printf("blockidx %d threadidx %d col %d, player count %d\n", blockIdx.x, threadIdx.x, col, playerCountInBlock);
  for (int tournamentStep=playerCountInBlock-2; tournamentStep>=0; tournamentStep--)
  {
    auto rowIndices = determineRowIndexPair(threadIdx.y, playerCountInBlock, tournamentStep);
    const int i = blockStart + rowIndices.first;
    const int j = blockStart + rowIndices.second;
    //printf("block id %d thread id %d blockstart %d i %d j %d \n", blockIdx.y, threadIdx.y, blockStart, i, j);

    if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
    {
      __syncthreads();
      continue;
    }

    const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    const scalar_t cij = C[thetaIndex];
    const scalar_t sij = S[thetaIndex];

    const int iOffset = i*N + col;
    const int jOffset = j*N + col;

    // Apply Givens: Update U's offsets
    const scalar_t Ui = U[iOffset];
    const scalar_t Uj = U[jOffset];

    U[iOffset] = Ui*cij - Uj*sij;
    U[jOffset] = Ui*sij + Uj*cij;

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
      printf("step %d :: just done with i %d and j %d on column %d\n", tournamentStep, i, j, col);
    }
    
  __syncthreads();
  }
}

template <typename scalar_t> __global__ void updateGivensElementsConveyorBelt(
  const scalar_t* const __restrict__ C,
  const scalar_t* const __restrict__ S,
  scalar_t* __restrict__ U,
  const int N,
  const int deadIndex,
  const int dMax,
  const int teamCount,
  const int tournamentStep)
{
  // If transpose k works on rows; otherwise on columns
  const int col = threadIdx.x + blockDim.x*blockIdx.x;
  if (col >= N)
  {
    return;
  }

  // rn 2*blockDepth-1 and ntilde-1 are eqaul, need to change what we pass to determineRowIndexPAir instead of ntilde
  auto rowIndices = determineRowIndexPair(blockIdx.y, teamCount, tournamentStep);
  
  const int playerCountPerTeam = blockDim.y;
  const int playerCountPerBlock = playerCountPerTeam * 2;
  const int i =  playerCountPerBlock * rowIndices.first + threadIdx.y; // home team
  const int iOffset = i*N + col;
  
  const int jStart = playerCountPerBlock * rowIndices.second + playerCountPerTeam;
  const int jEnd = jStart + playerCountPerTeam;
  int j = jStart + threadIdx.y;
  
  scalar_t Ui = U[iOffset];
  for (int step =0; step < playerCountPerTeam; step++, j++)
  {
    if (j >= jEnd)
    {
      j -= playerCountPerTeam;
    }
    
    if (areRowIndicesOutOfRange(i, j, deadIndex, dMax))
    {
      __syncthreads();
      continue;
    }

    const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    const scalar_t cij = C[thetaIndex];
    const scalar_t sij = S[thetaIndex];
    const int jOffset = j*N + col;

    // Apply Givens: Update U's offsets
    const scalar_t Uj = U[jOffset];

    Ui = Ui*cij - Uj*sij;
    U[jOffset] = Ui*sij + Uj*cij;


    if(blockIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 0)
    {
      printf("step %d -> thread %d, just done with i %d and j %d on column %d\n", step,threadIdx.y, i, j, col);

    }
    __syncthreads();
  }

  U[iOffset] = Ui;
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

torch::Tensor rotMatForwardCuda(torch::Tensor thetas, int64_t N, int threadsPerBlock)
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

  const int intraTeamTournamentBlockDepth = 4;
  const int interTeamTournamentBlockDepth = 8;

  int threadsPerRow = threadsPerBlock/intraTeamTournamentBlockDepth; // 128
  const dim3 threads(threadsPerRow, intraTeamTournamentBlockDepth); // 128, 4
  
  int nBlocksX = (N/threadsPerRow) + (N %threadsPerRow != 0); // 1
  int nBlocksY = ((Ntilde / 2)/intraTeamTournamentBlockDepth) + ((Ntilde / 2) % intraTeamTournamentBlockDepth != 0); // 2
  const dim3 blocks(nBlocksX, nBlocksY);

  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel. There are n-1 steps
  AT_DISPATCH_FLOATING_TYPES(
    thetas.type(),
    "rotMatForwardCuda",
    ([&]{ updateGivensElementsRoundRobin<scalar_t><<<blocks,threads>>>(
      C.data_ptr<scalar_t>(),
      S.data_ptr<scalar_t>(),
      U.data_ptr<scalar_t>(),
      N, deadIndex, dMax);}));
  
  cudaDeviceSynchronize();
  cout << "\n\n"; 
  int stepsLeft = Ntilde - (intraTeamTournamentBlockDepth * 2);
  if (stepsLeft <= 0 )
  {
    return U;
  }

  threadsPerRow = threadsPerBlock/interTeamTournamentBlockDepth;
  const dim3 threads2(threadsPerRow, interTeamTournamentBlockDepth);
  
  const int nBlocksX2 = (N/threadsPerRow) + (N%threadsPerRow != 0);
  const int nBlocksY2 = ((Ntilde / 2)/interTeamTournamentBlockDepth) + ((Ntilde / 2)% interTeamTournamentBlockDepth != 0);
  const dim3 blocks2(nBlocksX2, nBlocksY2);

  const int teamCount = (Ntilde/interTeamTournamentBlockDepth) + (Ntilde%interTeamTournamentBlockDepth != 0);
  for (int tournamentStep=teamCount-2; tournamentStep>=0; tournamentStep--)
  {
    if (tournamentStep == 1 && 
        interTeamTournamentBlockDepth == intraTeamTournamentBlockDepth)
    {
      continue;
    }

    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatForwardCuda",
      ([&]{ updateGivensElementsConveyorBelt<scalar_t><<<blocks2,threads2>>>(
        C.data_ptr<scalar_t>(),
        S.data_ptr<scalar_t>(),
        U.data_ptr<scalar_t>(),
        N, deadIndex, dMax, teamCount, tournamentStep);}));
  }

  return U;
}

template <typename scalar_t> __global__ void setA(
  const scalar_t* const __restrict__ C,
  const scalar_t* const __restrict__ S,
  scalar_t* __restrict__ M,
  scalar_t* __restrict__ UfTrans,
  scalar_t* __restrict__ A,
  const size_t N,
  const int deadIndex,
  const size_t Ntilde,
  const int dMax,
  const int tournamentStep)
{
  // k is the column index of M and the row index of Uf, to set col of A
  const int k = threadIdx.y + blockDim.y*blockIdx.y;
  if (k >= N)
  {
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
  const scalar_t cij = C[thetaIndex];
  const scalar_t sij = S[thetaIndex];

  // Apply Givens: Update U's offsets
  const int ik = i*N + k;
  const int jk = j*N + k;

  const scalar_t Ui = UfTrans[ik];
  const scalar_t Uj = UfTrans[jk];

  const scalar_t newUfTik = Ui*cij - Uj*sij;
  UfTrans[ik] = newUfTik;

  const scalar_t newUfTjk = Ui*sij + Uj*cij;
  UfTrans[jk] = newUfTjk;

  // Repeat for M
  const scalar_t Mi = M[ik];
  const scalar_t Mj = M[jk];

  const scalar_t newMik = Mi*cij - Mj*sij;
  M[ik] = newMik;

  const scalar_t newMjk = Mi*sij + Mj*cij;
  M[jk] = newMjk;

  // Set A, skip a write if possible can
  const scalar_t val = newMik * newUfTjk - newMjk * newUfTik;
  if(val != 0)
  {
    A[k + blockIdx.x * N] = val;
  }
}

template <typename scalar_t> __global__ void setJVP(
  const scalar_t* __restrict__ Asum,
  scalar_t* __restrict__ JVP,
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

torch::Tensor rotMatBackwardCuda(
  torch::Tensor thetas,
  torch::Tensor U,
  torch::Tensor G,
  int threadsPerBlock)
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

  // Note 0 dim is Ntilde not N
  auto A = torch::zeros({Ntilde/2,N}, tensOptions).detach();

  // Replaced to deal with frame setting
  // auto M = G.detach().t().contiguous();
  auto M = torch::zeros({N,N}, tensOptions);
  M.index_put_({Slice(0,G.size(1)), Slice()}, G.detach().t());

  // auto UfTrans = torch::clone(U.t());
  auto UfTrans = U.t().contiguous();

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  // CUDA grid: blocks of size: Ntilde/2 x ceil(N/nThreads)
  const int nBlocksX = Ntilde / 2;
  const int nBlocksY = N/threadsPerBlock + (N % threadsPerBlock != 0);

  const dim3 blocks(nBlocksX, nBlocksY);
  const dim3 threads(1, threadsPerBlock);

  cudaStream_t stream1;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  // The circle-method is used to generate round-robin sequences per block (equivalent to scheduling round-robin sports tournaments)
  // 'tournamentStep' refers to to the current turn of the tournament, where all updates are executed in parallel.
  // There are n-1 steps
  for (int tournamentStep=Ntilde-2; tournamentStep>=0; tournamentStep--)
  {
    // Set A els
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatBackwardCuda",
      ([&]{ setA<scalar_t><<<blocks,threads>>>(
        C.data_ptr<scalar_t>(),
        S.data_ptr<scalar_t>(),
        M.data_ptr<scalar_t>(),
        UfTrans.data_ptr<scalar_t>(),
        A.data_ptr<scalar_t>(),
        N, deadIndex, Ntilde, dMax, tournamentStep);}));

    // Sum rows
    cudaStreamSynchronize(stream1);
    auto Asum = A.sum(1);

    // Parallelized setting of JVP els from Asum; note here Ntilde/2 blocks, 1 thr/blck
    AT_DISPATCH_FLOATING_TYPES(
      thetas.type(),
      "rotMatBackwardCuda",
      ([&]{ setJVP<scalar_t><<<Ntilde/2,1,0,stream1>>>(Asum.data<scalar_t>(), JVP.data<scalar_t>(), N, deadIndex, Ntilde, dMax, tournamentStep);}));
  }
  
  cudaStreamSynchronize(stream1);
  return JVP;
}
