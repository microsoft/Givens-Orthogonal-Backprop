// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include <functional>
#include <vector>

//using namespace torch::indexing;

#define MaximumThreadsPerBlock 1024
#define WarpSize 32

#define ThreadsPerRowForward 64
#define ThreadsPerRowBackward 128

// DELETE FORWARD SUFFIX, SAME FOR FORWARD AND BACKWARD
#define InterTeamRRThreadsPerBlock MaximumThreadsPerBlock  /2
#define InterTeamBlockDepth (int)(InterTeamRRThreadsPerBlock/WarpSize)

// Current implementation dictates InterTeamBlockDepth to be 2 *IntraTeamBlockDepth
#define IntraTeamRRThreadsPerBlock MaximumThreadsPerBlock / 4
#define IntraTeamBlockDepth (int)(InterTeamBlockDepth/2)

// Compile time constants dictate that minimum width must be 32. This is also optimal for global memory broadcast behavior
#define InterTeamBlockWidth (int)(InterTeamRRThreadsPerBlock/InterTeamBlockDepth)
#define IntraTeamBlockWidth (int)(IntraTeamRRThreadsPerBlock/IntraTeamBlockDepth)

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

/************************* HELPER FUNCTIONS *********************************/

int getTeamSizeForTesting()
{
  return InterTeamBlockDepth;
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
  const int firstDummyIndex, 
  const int dMax)
{
  // check if the coordinates are out of range or equal dummy coordinate (dummy exists when N is odd)
  return j >= firstDummyIndex || (i > dMax && j > dMax);
}

void incrementIfNotEven(int &Ntilde)
{
  if (Ntilde % 2 != 0)
  {
    Ntilde += 1;
  }
}

std::tuple<int, int, int> determineRotMatConstants(const int &nThetas, const int &N)
{
  auto dMax = N-1; // If nThetas == maxPairs
  if (nThetas < N*(N-1)/2)
  {
    dMax -= 1 + int(sqrt(1 - 4*(2*nThetas - N*(N-1)))) / 2;
  }

  // Handle odd N; in that case Ntilde is the even augmented dimension
  int Ntilde = N;
  const int dummyIndex = Ntilde;
  incrementIfNotEven(Ntilde);

  return std::make_tuple(dMax, dummyIndex, Ntilde);
}

const dim3 prepareBlocksForTournament(const int &B, const int &Ntilde, const int &width, const int &depth )
{
  const int nBlocksX = (B/width) + (B % width != 0);
  
  const int rotationsPerRound = Ntilde/2;
  const int nBlocksY = (rotationsPerRound / depth) + (rotationsPerRound % depth != 0); 
  
  const dim3 blocks(nBlocksX, nBlocksY);
  return blocks;
}

template <typename scalar_t, unsigned int blockSize>
  __device__ void blockReduce(
    volatile scalar_t* sA, 
    const int &tid)
{
  if (blockSize == 1024) { if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) {if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
  if (blockSize >= 64) { if (tid < 32 ) sA[tid] += sA[tid + 32]; __syncthreads(); }
}

template <typename scalar_t, unsigned int blockSize>
  __device__ void warpReduce(
    volatile scalar_t* sdata, 
    const int &tid)
{
  __syncwarp();
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
  __device__ void convergeWithBlock(const int &tid)
{
  if (blockSize == 1024) { if (tid < 512)  __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256)   __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128)   __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64)  __syncthreads(); }
  if (blockSize >= 64) { if (tid < 32 )  __syncthreads(); }
}

/************************* FORWARD PROPAGATION*******************************/
/****** USING THE CIRCLE METHOD FOR GENERATING THE ROUND ROBIN SEQUENCE *****/

 template <typename scalar_t>  
  __global__ void ApplyRoundRobinGivensRotationMatrix(
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> U,
    const int dummyIndex,
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

  if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
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

torch::Tensor rotMatForwardCuda(torch::Tensor X, torch::Tensor thetas)
{
  const int N = X.size(0);
  auto constants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);

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
          dummyIndex,Ntilde, dMax, tournamentStep);
      }));
  }

  return X;
}

/************************* BACKWARD PROPAGATION******************************/
// USING THE CIRCLE ROUND ROBIN TOURNAMENT FOR SEQUENCING GIVENS ROTATIONS

template <typename scalar_t> 
  __global__ void CalculateRoundRobinGivensThetaJVPs(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UX,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> G,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> JVP,
    const int dummyIndex,
    const int Ntilde,
    const int dMax,
    const int tournamentStep)
{
  __shared__ scalar_t sA[ThreadsPerRowBackward];

  // k is the column index of M and the row index of Uf, to set col of A
  const int tid = threadIdx.x;
  const int k = tid + blockDim.y*blockIdx.y;
  if (k >= UX.size(1))
  {
    sA[tid] = 0;
    return;
  }

  auto rowIndices = determineRowIndexPair(blockIdx.x, Ntilde, tournamentStep);
  int i = rowIndices.first;
  int j = rowIndices.second;

  if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
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
  blockReduce<scalar_t, ThreadsPerRowBackward>(sA, tid);
  if(tid < 16) warpReduce<scalar_t, ThreadsPerRowBackward>(sA, tid);
  if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
}

std::pair<torch::Tensor, torch::Tensor> rotMatBackwardCuda(
  torch::Tensor thetas,
  torch::Tensor UX,
  torch::Tensor G)
{
  auto constants = determineRotMatConstants(thetas.size(0), UX.size(0));
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  auto thetasTensorOptions = torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());
  auto JVP = torch::zeros_like(thetas, thetasTensorOptions);

  auto B = UX.size(1);
  const int nBlocksY = B/ThreadsPerRowBackward + (B % ThreadsPerRowBackward != 0);
  const dim3 blocks(Ntilde / 2, nBlocksY);
  const dim3 threads(ThreadsPerRowBackward,1);

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
          dummyIndex, Ntilde, dMax, tournamentStep);
      }));
  }
  
  return std::make_pair(G, JVP);
}

/************************* FORWARD PROPAGATION*******************************/
// USING THE TEAM ROUND ROBIN TOURNAMENT FOR SEQUENCING GIVENS ROTATIONS

template <typename scalar_t>
  __global__ void PlayIntraTeamTournament(
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const int Ntilde,
    const int dummyIndex,
    const int dMax)
{
  // If transpose k works on rows; otherwise on columns
  const int tid = threadIdx.x;
  const int tidY = threadIdx.y;
  const int k = tid + blockDim.x*blockIdx.x; 

  if (k >= X.size(1))
  {
    return;
  }

  int playerCountInBlock = IntraTeamBlockDepth *2;
  int blockStart = playerCountInBlock * blockIdx.y;
  if (playerCountInBlock > Ntilde-blockStart)
  {
    playerCountInBlock = Ntilde-blockStart;
  }

  if (tidY*2 >= playerCountInBlock)
  {
    return;
  }

  const int N = X.size(0);
  int i, j, thetaIndex;
  scalar_t cij, sij, Xi, Xj;
  
  for (int tournamentStep=0; tournamentStep<=playerCountInBlock-2; tournamentStep++)
  {
    auto rowIndices = determineRowIndexPair(tidY, playerCountInBlock, tournamentStep);
    i = blockStart + rowIndices.first;
    j = blockStart + rowIndices.second;
    
    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      __syncthreads();
      continue;
    }

    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = S[thetaIndex];
    
    Xi = X[i][k];
    Xj = X[j][k];

    X[i][k] = Xi*cij - Xj*sij;
    X[j][k] = Xi*sij + Xj*cij;
    __syncthreads();
  }
}

template <typename scalar_t> __global__ void PlayTeamTournamentMatch(
  at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
  at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
  at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
  const int dummyIndex,
  const int dMax,
  const int teamCount,
  const int dummyTeamIndex,
  const int tournamentStep)
{
  // If transpose k works on rows; otherwise on columns
  const int col = threadIdx.x + blockDim.x*blockIdx.x;
  if (col >= X.size(1))
  {
    return;
  }

  auto matchedTeams = determineRowIndexPair(blockIdx.y, teamCount, tournamentStep);
  if (matchedTeams.second == dummyTeamIndex)  return;

  const int playerCountPerTeam = blockDim.y;
  const int i =  playerCountPerTeam * matchedTeams.first + threadIdx.y; // home team
  scalar_t Xi = X[i][col];
  
  const int jStart = playerCountPerTeam * matchedTeams.second; //visiting team
  const int jEnd = jStart + playerCountPerTeam;
  int j = jStart + threadIdx.y;
  
  const int N = X.size(0);
  int thetaIndex;
  scalar_t Xj, sij, cij;
  for (int step =0; step < playerCountPerTeam; step++, j++)
  {
    if (j >= jEnd)
    {
      j -= playerCountPerTeam;
    }
    
    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      __syncthreads();
      continue;
    }

    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = S[thetaIndex];

    // Apply Givens: Update U's offsets
    Xj = X[j][col];

    X[j][col] = Xi*sij + Xj*cij;
    Xi = Xi*cij - Xj*sij;
    __syncthreads();
  }

  X[i][col] = Xi;
}

bool ScheduleIntraTeamTournaments(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor X,
  std::tuple<int, int, int> constants)
{
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);

  const dim3 blocks = prepareBlocksForTournament(X.size(1), Ntilde, IntraTeamBlockWidth, IntraTeamBlockDepth);
  const dim3 threads(IntraTeamBlockWidth, IntraTeamBlockDepth);

  AT_DISPATCH_FLOATING_TYPES(
    C.type(),
    "rotMatForwardCuda",
    ([&]{ PlayIntraTeamTournament<scalar_t><<<blocks,threads>>>(
      C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      X.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
      Ntilde, dummyIndex, dMax);}));

  return blocks.y == 1;
}

void ScheduleInterTeamTournament(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor X,
  std::tuple<int, int, int> constants)
{
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);
  
  const dim3 blocks = prepareBlocksForTournament(X.size(1), Ntilde, InterTeamBlockWidth, InterTeamBlockDepth);
  const dim3 threads(InterTeamBlockWidth, InterTeamBlockDepth);

  int teamCount = (Ntilde/InterTeamBlockDepth) + (Ntilde%InterTeamBlockDepth != 0);
  const int dummyTeamIndex = teamCount;
  incrementIfNotEven(teamCount);
  
  for (int tournamentStep=teamCount-2; tournamentStep>=0; tournamentStep--)
  {
    AT_DISPATCH_FLOATING_TYPES(
      C.type(),
      "rotMatForwardCuda",
      ([&]{ PlayTeamTournamentMatch<scalar_t><<<blocks,threads>>>(
        C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        X.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
        dummyIndex, dMax, teamCount, dummyTeamIndex, tournamentStep);}));
  }
}

torch::Tensor rotMatForwardCudaTeamRR(torch::Tensor X, torch::Tensor thetas)
{
  auto constants = determineRotMatConstants(thetas.size(0), X.size(0));
  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());
  
  bool allThetasFitToOneTeam = ScheduleIntraTeamTournaments(C, S, X, constants);
  if (allThetasFitToOneTeam) return X;
  ScheduleInterTeamTournament(C, S, X, constants);

  return X;
}

/************************* BACKWARD PROPAGATION*******************************/
// USING THE TEAM ROUND ROBIN TOURNAMENT FOR SEQUENCING GIVENS ROTATIONS

template <typename scalar_t> 
  __global__ void PlayTeamTournamentMatchForThetaGrad(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UX,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> G,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> JVP,
    const int dummyIndex,
    const int dMax,
    const int teamCount,
    const int dummyTeamIndex,
    const int tournamentStep)
{
  __shared__ scalar_t sAGridForBlock[InterTeamBlockDepth][InterTeamBlockWidth];
  scalar_t* sA = sAGridForBlock[threadIdx.y];

  // If transpose k works on rows; otherwise on columns
  const int k = threadIdx.x + blockDim.x*blockIdx.x;
  const int tid = threadIdx.x;
  const int B = UX.size(1);
  if (k >= B) // do we need a tidyY check here?
  {
    sA[tid] = 0;
    return;
  }

  auto matchedTeams = determineRowIndexPair(blockIdx.y, teamCount, tournamentStep);
  if (matchedTeams.second == dummyTeamIndex)  return;

  const int playerCountPerTeam = blockDim.y;
  const int i =  playerCountPerTeam * matchedTeams.first + threadIdx.y; // home team
  scalar_t UXi = UX[i][k];
  scalar_t Gi = G[i][k];

  const int jStart = playerCountPerTeam * matchedTeams.second; //visiting team
  int j = jStart + threadIdx.y - 1;
  
  const int N = UX.size(0);
  int thetaIndex;
  scalar_t cij, sij, UXj, newUXj, Gj, newGj;
  for (int step =0; step < playerCountPerTeam; step++, j--)
  {
     __syncthreads();
    if (j < jStart)
    {
      j += playerCountPerTeam;
    }
    
    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      sA[tid] = 0;
      __syncthreads();
      convergeWithBlock<InterTeamBlockWidth>(tid);
      continue;
    }

    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = -1 * S[thetaIndex];

    // Apply Givens: Update U's offsets
    UXj = UX[j][k];
    newUXj = UXi*sij + UXj*cij;

    Gj = G[j][k];
    newGj = Gi*sij + Gj*cij; 

    UXi = UXi*cij - UXj*sij; 
    Gi = Gi*cij - Gj*sij;

    UX[j][k] = newUXj; 
    G[j][k] = newGj;

    auto res = UXi * newGj - newUXj * Gi;
    sA[tid] = res;
    __syncthreads();

    // Reduce
    blockReduce<scalar_t, InterTeamBlockWidth>(sA, tid);
    if(tid < 16) warpReduce<scalar_t, InterTeamBlockWidth>(sA, tid);
    if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
  }

  UX[i][k] = UXi;
  G[i][k] = Gi;
}

template <typename scalar_t> 
  __global__ void PlayIntraTeamTournamentsForThetaGrad(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UX,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> G,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> JVP,
    const int Ntilde,
    const int dummyIndex,
    const int dMax)
{
  __shared__ scalar_t sAGridForBlock[IntraTeamBlockDepth][IntraTeamBlockWidth];
  const int tidY = threadIdx.y;
  scalar_t* sA = sAGridForBlock[tidY];
  
  // k is the column index of M and the row index of Uf, to set col of A
  const int tid = threadIdx.x;
  const int k = tid + blockDim.x*blockIdx.x; 
  if (k >= UX.size(1))
  {
    sA[tid] = 0;
    return;
  }

  int playerCountInBlock = IntraTeamBlockDepth *2;
  int blockStart = playerCountInBlock * blockIdx.y;
  if (playerCountInBlock > Ntilde-blockStart)
  {
    playerCountInBlock = Ntilde-blockStart;
  }

  if (tidY*2 >= playerCountInBlock)
  {
    return;
  }

  const int N = UX.size(0);
  int i, j, thetaIndex;
  scalar_t cij, sij, UXi, UXj, newUXi, newUXj, Gi, Gj, newGi, newGj;

  for (int tournamentStep=playerCountInBlock-2; tournamentStep>=0; tournamentStep--)
  {
    __syncthreads();
    auto rowIndices = determineRowIndexPair(tidY, playerCountInBlock, tournamentStep);
    i = blockStart + rowIndices.first;
    j = blockStart + rowIndices.second;

    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      sA[tid] = 0;
      __syncthreads();
      convergeWithBlock<IntraTeamBlockWidth>(tid);
      continue;
    }
    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = -1 * S[thetaIndex]; // Transpose of a Givens rotation has the signs of sij flipped

    Gi = G[i][k];
    Gj = G[j][k];

    newGi = Gi*cij - Gj*sij;
    G[i][k] = newGi;

    newGj = Gi*sij + Gj*cij;
    G[j][k] = newGj;

    UXi = UX[i][k];
    UXj = UX[j][k];

    newUXi = UXi*cij - UXj*sij;
    UX[i][k] = newUXi;

    newUXj = UXi*sij + UXj*cij;
    UX[j][k] = newUXj;

    sA[tid] = newUXi * newGj - newUXj * newGi;
    __syncthreads();

    // Reduce
    blockReduce<scalar_t, IntraTeamBlockWidth>(sA, tid);
    if(tid < 16) warpReduce<scalar_t, IntraTeamBlockWidth>(sA, tid);
    if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
  }
}

void ScheduleInterTeamTournamentForThetaGrads(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor UX,
  torch::Tensor G,
  torch::Tensor JVP,
  std::tuple<int, int, int> constants)
{
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);
  
  int teamCount = (Ntilde / InterTeamBlockDepth) + (Ntilde % InterTeamBlockDepth != 0);
  if (teamCount == 1)
  {
    return;
  }
  const int dummyTeamIndex = teamCount;
  incrementIfNotEven(teamCount);

  const dim3 blocks = prepareBlocksForTournament(UX.size(1), Ntilde, InterTeamBlockWidth, InterTeamBlockDepth);
  const dim3 threads(InterTeamBlockWidth, InterTeamBlockDepth);
  
  for (int tournamentStep=0; tournamentStep<=teamCount-2; tournamentStep++)
  {
    AT_DISPATCH_FLOATING_TYPES(
      C.type(),
      "rotMatForwardCuda",
      ([&]{ PlayTeamTournamentMatchForThetaGrad<scalar_t><<<blocks,threads>>>(
        UX.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
        G.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
        C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        JVP.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        dummyIndex, dMax, teamCount, dummyTeamIndex, tournamentStep);}));
  }
}

void ScheduleIntraTeamTournamentsForThetaGrads(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor UX,
  torch::Tensor G,
  torch::Tensor JVP,
  std::tuple<int, int, int> constants)
{
  auto dMax = std::get<0>(constants);
  auto dummyIndex = std::get<1>(constants);
  auto Ntilde = std::get<2>(constants);

  const dim3 blocks = prepareBlocksForTournament(UX.size(1), Ntilde, IntraTeamBlockWidth, IntraTeamBlockDepth);
  const dim3 threads(IntraTeamBlockWidth, IntraTeamBlockDepth);

  AT_DISPATCH_FLOATING_TYPES(
    C.type(),
    "rotMatForwardCuda",
    ([&]{ PlayIntraTeamTournamentsForThetaGrad<scalar_t><<<blocks,threads>>>(
      UX.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
      G.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
      C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      JVP.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      Ntilde, dummyIndex, dMax);}));
}

std::pair<torch::Tensor, torch::Tensor> rotMatBackwardCudaTeamRR(
  torch::Tensor thetas,
  torch::Tensor UX,
  torch::Tensor G)
{
  auto constants = determineRotMatConstants(thetas.size(0), UX.size(0));
  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  auto thetasTensorOptions = torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());
  auto JVP = torch::zeros_like(thetas, thetasTensorOptions);

  ScheduleInterTeamTournamentForThetaGrads(C, S, UX, G, JVP, constants);
  ScheduleIntraTeamTournamentsForThetaGrads(C, S, UX, G, JVP, constants);

  return std::make_pair(G, JVP);
}
