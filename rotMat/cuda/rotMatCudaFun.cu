// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include <functional>
#include <vector>

using namespace torch::indexing;

#define MaximumThreadsPerBlock 1024
#define WarpSize 32

#define ThreadsPerRowForward 128
#define ThreadsPerRowBackward 256

#define InterTeamRRThreadsPerBlockForward MaximumThreadsPerBlock
#define InterTeamBlockDepthForward (int)(InterTeamRRThreadsPerBlockForward/256)
#define InterTeamBlockWidthForward (int)(InterTeamRRThreadsPerBlockForward/InterTeamBlockDepthForward)


// Current implementation dictates InterTeamBlockDepthForward to be 2 *IntraTeamBlockDepthForward
#define IntraTeamRRThreadsPerBlockForward (int)(MaximumThreadsPerBlock)
#define IntraTeamBlockDepthForward (int)(InterTeamBlockDepthForward/2)
#define IntraTeamBlockWidthForward (int)(IntraTeamRRThreadsPerBlockForward/IntraTeamBlockDepthForward)


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
  return InterTeamBlockDepthForward;
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

void addDummyIndexIfNotEven(int &Ntilde, int &dummyIndex)
{
  dummyIndex = Ntilde;
  if (Ntilde % 2 != 0)
  {
    Ntilde += 1;
    dummyIndex = Ntilde-1;
  }
}

std::tuple<int, int, int> determineRotMatConstants(const int nThetas, const int N)
{
  auto dMax = N-1; // If nThetas == maxPairs
  if (nThetas < N*(N-1)/2)
  {
    dMax -= 1 + int(sqrt(1 - 4*(2*nThetas - N*(N-1)))) / 2;
  }

  // Handle odd N; in that case Ntilde is the even augmented dimension
  int dummyIndex;
  int Ntilde = N;
  addDummyIndexIfNotEven(Ntilde, dummyIndex);

  return std::make_tuple(dMax, dummyIndex, Ntilde);
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
  __device__ void warpReduceAtBackwardIntraTeamRR(
    volatile scalar_t* sdata,
    int tid)
{
  if (IntraTeamRRThreadsPerBlockForward >= 64)  sdata[tid] += sdata[tid + 32];
  if (IntraTeamRRThreadsPerBlockForward >= 32) sdata[tid] += sdata[tid + 16];
  if (IntraTeamRRThreadsPerBlockForward >= 16) sdata[tid] += sdata[tid + 8];
  if (IntraTeamRRThreadsPerBlockForward >= 8) sdata[tid] += sdata[tid + 4];
  if (IntraTeamRRThreadsPerBlockForward >= 4) sdata[tid] += sdata[tid + 2];
  if (IntraTeamRRThreadsPerBlockForward >= 2) sdata[tid] += sdata[tid + 1];
}

template <typename scalar_t>
  __device__ void warpReduceAtBackwardInterTeamRR(
    volatile scalar_t* sdata, 
    int tid)
{
  if (InterTeamRRThreadsPerBlockForward >= 64)  sdata[tid] += sdata[tid + 32];
  if (InterTeamRRThreadsPerBlockForward >= 32) sdata[tid] += sdata[tid + 16];
  if (InterTeamRRThreadsPerBlockForward >= 16) sdata[tid] += sdata[tid + 8];
  if (InterTeamRRThreadsPerBlockForward >= 8) sdata[tid] += sdata[tid + 4];
  if (InterTeamRRThreadsPerBlockForward >= 4) sdata[tid] += sdata[tid + 2];
  if (InterTeamRRThreadsPerBlockForward >= 2) sdata[tid] += sdata[tid + 1];
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
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());

  const int nBlocksY = X.size(0)/ThreadsPerRowForward + (X.size(0)% ThreadsPerRowForward != 0);
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

std::pair<torch::Tensor, torch::Tensor> rotMatBackwardCuda(
  torch::Tensor thetas,
  torch::Tensor UX,
  torch::Tensor G)
{
  auto N = UX.size(0);
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
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
          dummyIndex, Ntilde, dMax, tournamentStep);
      }));
  }
  
  return std::make_pair(G, JVP);
}

/************************* FORWARD PROPAGATION*******************************/
// USING THE TEAM ROUND ROBIN TOURNAMENT FOR SEQUENCING GIVENS ROTATIONS

template <typename scalar_t>
  __global__ void PlayIndividualTournamentWithinTeams(
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
  if (k >= X.size(1) || tidY*2 >= Ntilde)
  {
    return;
  }

  int playerCountInBlock = IntraTeamBlockDepthForward *2;
  int blockStart = playerCountInBlock * blockIdx.y;
  if (playerCountInBlock > Ntilde-blockStart)
  {
    playerCountInBlock = Ntilde-blockStart;
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
    //if( tid == 0 ) printf("%d %d got one! -> S %.6f C %.6f \n", i, j, sij, cij);

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

    const int thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    const scalar_t cij = C[thetaIndex];
    const scalar_t sij = S[thetaIndex];

    // Apply Givens: Update U's offsets
    const scalar_t Xj = X[j][col];

    //if( col == 0 ) printf("%d %d got one! ->  S %.6f C %.6f \n ", i, j, sij, cij);

    // must update uj before updating ui
    X[j][col] = Xi*sij + Xj*cij;
    Xi = Xi*cij - Xj*sij;

    __syncthreads();
  }

  X[i][col] = Xi;
}


bool ScheduleIndividualTournamentsWithinTeams(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor X,
  std::tuple<int, int, int> rotMatConstants)
{
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  const int threadsWithIdenticalWork = IntraTeamRRThreadsPerBlockForward/IntraTeamBlockDepthForward;
  const dim3 threads(threadsWithIdenticalWork, IntraTeamBlockDepthForward);

  const int B = X.size(1);
  const int nBlocksX = (B/threadsWithIdenticalWork) + (B %threadsWithIdenticalWork != 0);
  const int nBlocksY = ((Ntilde / 2)/IntraTeamBlockDepthForward) + ((Ntilde / 2) % IntraTeamBlockDepthForward != 0); 
  const dim3 blocks(nBlocksX, nBlocksY);

  AT_DISPATCH_FLOATING_TYPES(
    C.type(),
    "rotMatForwardCuda",
    ([&]{ PlayIndividualTournamentWithinTeams<scalar_t><<<blocks,threads>>>(
      C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
      X.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
      Ntilde, dummyIndex, dMax);}));

  return nBlocksY == 1;
}

void ScheduleTeamTournament(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor X,
  std::tuple<int, int, int> rotMatConstants)
{
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  const int threadsWithIdenticalWork = InterTeamRRThreadsPerBlockForward/InterTeamBlockDepthForward;
  const dim3 threads(threadsWithIdenticalWork, InterTeamBlockDepthForward);
  
  const int B = X.size(1);
  const int nBlocksX = (B/threadsWithIdenticalWork) + (B%threadsWithIdenticalWork != 0);
  const int nBlocksY = ((Ntilde / 2)/InterTeamBlockDepthForward) + ((Ntilde / 2)% InterTeamBlockDepthForward != 0);
  const dim3 blocks(nBlocksX, nBlocksY);

  int teamCount = (Ntilde/InterTeamBlockDepthForward) + (Ntilde%InterTeamBlockDepthForward != 0);
  int dummyTeamIndex = -1;
  addDummyIndexIfNotEven(teamCount,dummyTeamIndex);
  
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
  auto rotMatConstants = determineRotMatConstants(thetas.size(0), X.size(0));
  auto C = torch::cos(thetas.detach());
  auto S = torch::sin(thetas.detach());
  
  bool allThetasFitToOneTeam = ScheduleIndividualTournamentsWithinTeams(C, S, X, rotMatConstants);
  if (allThetasFitToOneTeam) return X;
  ScheduleTeamTournament(C, S, X, rotMatConstants);
  
  return X;
}

/************************* BACKWARD PROPAGATION*******************************/
// USING THE TEAM ROUND ROBIN TOURNAMENT FOR SEQUENCING GIVENS ROTATIONS

template <typename scalar_t> 
  __global__ void PlayThetaGradTeamTournamentMatch(
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
  __shared__ scalar_t sAPerRow[IntraTeamBlockDepthForward][IntraTeamBlockWidthForward];
  scalar_t* sA = sAPerRow[threadIdx.y];

  // If transpose k works on rows; otherwise on columns
  const int k = threadIdx.x + blockDim.x*blockIdx.x;
  const int tid = threadIdx.x;
  if (k >= UX.size(1))
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
  const int jEnd = jStart + playerCountPerTeam;
  int j = jStart + threadIdx.y;
  
  const int N = UX.size(0);
  int thetaIndex;
  scalar_t cij, sij, UXj, newUXi, newUXj, Gj, newGi, newGj;
  for (int step =0; step < playerCountPerTeam; step++, j++)
  {
    if (j >= jEnd)
    {
      j -= playerCountPerTeam;
    }
    
    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      sA[tid] = 0;
      __syncthreads();
      continue;
    }

    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = -S[thetaIndex];

    // Apply Givens: Update U's offsets
    UXj = UX[j][k];
    newUXj = UXi*sij + UXj*cij; // must update j before updating i
    UXi = UXi*cij - UXj*sij; 

    Gj = G[j][k];
    newGj = Gi*sij + Gj*cij; // must update j before updating i
    Gi = Gi*cij - Gj*sij;

    UX[j][k] = newUXj; 
    G[j][k] = newGj;

    sA[tid] = newUXi * newGj - newUXj * newGi;
    __syncthreads();

    // Reduce
    if (InterTeamBlockWidthForward == 1024) {
      if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads(); }
    if (InterTeamBlockWidthForward >= 512) {
      if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads(); }
    if (InterTeamBlockWidthForward >= 256) {
      if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
    if (InterTeamBlockWidthForward >= 128) {
      if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduceAtBackwardInterTeamRR(sA, tid);
    
    if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
  }

  UX[i][k] = UXi;
}


template <typename scalar_t> 
  __global__ void PlayThetaGradTournamentWithinTeams(
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> UX,
    at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> G,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> C,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> S,
    at::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> JVP,
    const int dummyIndex,
    const int Ntilde,
    const int dMax)
{
  __shared__ scalar_t sAPerRow[IntraTeamBlockDepthForward][IntraTeamBlockWidthForward];
  
  const int tidY = threadIdx.y;
  scalar_t* sA = sAPerRow[tidY];

  // k is the column index of M and the row index of Uf, to set col of A
  const int tid = threadIdx.x;
  const int k = tid + blockDim.x*blockIdx.x; 
  if (k >= UX.size(1) || k*2 >= Ntilde)
  {
    sA[tid] = 0;
    return;
  }

  int playerCountInBlock = IntraTeamBlockDepthForward *2;
  int blockStart = playerCountInBlock * blockIdx.y;

  if (playerCountInBlock > Ntilde-blockStart)
  {
    playerCountInBlock = Ntilde-blockStart;
  }

  const int N = UX.size(0);
  int i, j, thetaIndex;
  scalar_t cij, sij, UXi, UXj, newUXi, newUXj, Gi, Gj, newGi, newGj;
  for (int tournamentStep=playerCountInBlock-2; tournamentStep>=0; tournamentStep--)
  {
    auto rowIndices = determineRowIndexPair(tidY, playerCountInBlock, tournamentStep);
    i = blockStart + rowIndices.first;
    j = blockStart + rowIndices.second;

    if (areRowIndicesOutOfRange(i, j, dummyIndex, dMax))
    {
      sA[tid] = 0;
      __syncthreads();
      continue;
    }

    thetaIndex = i*N - (i+2)*(i+1)/2 + j;
    cij = C[thetaIndex];
    sij = -S[thetaIndex]; // Transpose of a Givens rotation has the signs of sij flipped
    
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
    if (IntraTeamBlockWidthForward == 1024) {
      if (tid < 512) { sA[tid] += sA[tid + 512]; } __syncthreads(); }
    if (IntraTeamBlockWidthForward >= 512) {
      if (tid < 256) { sA[tid] += sA[tid + 256]; } __syncthreads(); }
    if (IntraTeamBlockWidthForward >= 256) {
      if (tid < 128) { sA[tid] += sA[tid + 128]; } __syncthreads(); }
    if (IntraTeamBlockWidthForward >= 128) {
      if (tid < 64) { sA[tid] += sA[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduceAtBackwardIntraTeamRR(sA, tid);
    
    if (tid == 0)  atomicAdd(&JVP[thetaIndex], sA[tid]);
  }
}


void ScheduleTeamTournamentForThetaGrads(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor UX,
  torch::Tensor G,
  torch::Tensor JVP,
  std::tuple<int, int, int> rotMatConstants)
{
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);
  
  const int rotationCountPerRound = Ntilde/2;
  bool allThetasFitToOneTeam = (rotationCountPerRound/IntraTeamBlockDepthForward) + (rotationCountPerRound % IntraTeamBlockDepthForward != 0);

  if (allThetasFitToOneTeam) {return;}

  const int threadsWithIdenticalWork = InterTeamRRThreadsPerBlockForward/InterTeamBlockDepthForward;
  const dim3 threads(threadsWithIdenticalWork, InterTeamBlockDepthForward);
  
  const int B = UX.size(1);
  const int nBlocksX = (B / threadsWithIdenticalWork) + (B % threadsWithIdenticalWork != 0);
  const int nBlocksY = (rotationCountPerRound / InterTeamBlockDepthForward) + (rotationCountPerRound % InterTeamBlockDepthForward != 0);
  const dim3 blocks(nBlocksX, nBlocksY);

  int teamCount = (Ntilde / InterTeamBlockDepthForward) + (Ntilde % InterTeamBlockDepthForward != 0);
  int dummyTeamIndex = -1;
  addDummyIndexIfNotEven(teamCount,dummyTeamIndex);
  
  for (int tournamentStep=0; tournamentStep<=teamCount-2; tournamentStep--)
  {
    AT_DISPATCH_FLOATING_TYPES(
      C.type(),
      "rotMatForwardCuda",
      ([&]{ PlayThetaGradTeamTournamentMatch<scalar_t><<<blocks,threads>>>(
        UX.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
        G.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
        C.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        S.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        JVP.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
        dummyIndex, dMax, teamCount, dummyTeamIndex, tournamentStep);}));
  }
}

void ScheduleIndividualTournamentsWithinTeamsForThetaGrads(
  torch::Tensor C, 
  torch::Tensor S, 
  torch::Tensor UX,
  torch::Tensor G,
  torch::Tensor JVP,
  std::tuple<int, int, int> rotMatConstants)
{
  auto dMax = std::get<0>(rotMatConstants);
  auto dummyIndex = std::get<1>(rotMatConstants);
  auto Ntilde = std::get<2>(rotMatConstants);

  const int threadsWithIdenticalWork = IntraTeamRRThreadsPerBlockForward/IntraTeamBlockDepthForward;
  const dim3 threads(threadsWithIdenticalWork, IntraTeamBlockDepthForward);
  
  const int B = UX.size(1);
  const int nBlocksX = (B/threadsWithIdenticalWork) + (B %threadsWithIdenticalWork != 0); // 1
  const int nBlocksY = ((Ntilde / 2)/IntraTeamBlockDepthForward) + ((Ntilde / 2) % IntraTeamBlockDepthForward != 0);
  const dim3 blocks(nBlocksX, nBlocksY);

  AT_DISPATCH_FLOATING_TYPES(
    C.type(),
    "rotMatForwardCuda",
    ([&]{ PlayThetaGradTournamentWithinTeams<scalar_t><<<blocks,threads>>>(
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

  ScheduleTeamTournamentForThetaGrads(C, S, UX, G, JVP, constants);
  ScheduleIndividualTournamentsWithinTeamsForThetaGrads(C, S, UX, G, JVP, constants);
  
  return std::make_pair(G, JVP);
}