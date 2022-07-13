// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include<torch/extension.h>
#include<vector>
#include<cmath>
#include <iostream>

// This version allows for a fixed nullspace mode: if the thetas are
// not a full upper triangular matrix, i.e. only the first "few" rows,
// the algorithm infers that all angles between pairs of coordinates
// in the left out rows are to be ignored as we assume there is no
// parametric control over them.

// Example: If N = 6, the thetas may correspond to:
// (01) (02) (03) (04) (05) (12) (13) (14) (15) (23) (24) (25)
// i.e. 12 parameters out of a maximum of 6*5/2 = 15, the routine
// infers that we have no parametric control over thetas (34) (35) (45).

using namespace torch::indexing;

std::pair<const int, const int> determineRowIndexPair(
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

bool areRowIndicesOutOfRange(const int i, const int j, const int deadIndex, const int dMax)
{
  // check if the coordinates are out of range or equal dummy coordinate (dummy exists when N is odd)
  return j == deadIndex || (i > dMax && j > dMax);
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

torch::Tensor rotMatFwd(torch::Tensor thetas, int64_t N)
{
	auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
	auto dMax = std::get<0>(rotMatConstants);
	auto deadVar = std::get<1>(rotMatConstants);
	auto Ntilde = std::get<2>(rotMatConstants);

	// auto U = torch::eye(N);
	// Set U same device and type as thetas
	auto tensOptions = torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());
	auto U = torch::eye(N, tensOptions);

	// Temp row variables, same type as U
	auto newRowUi = torch::zeros({ N }, tensOptions);
	auto newRowUj = torch::zeros({ N }, tensOptions);

	for (int64_t step = Ntilde - 2; step >= 0; step--)
	{
		for (int64_t blockIndx = 0; blockIndx < Ntilde / 2; blockIndx++)
		{
			auto rowIndices = determineRowIndexPair(blockIndx, Ntilde, step);
			int i = rowIndices.first;
			int j = rowIndices.second;

			if (areRowIndicesOutOfRange(i, j, deadVar, dMax))
			{
				continue;
			}

			int64_t thetaIndx = i * N - (i + 2) * (i + 1) / 2 + j;
			auto cij = torch::cos(thetas[thetaIndx]).item();
			auto sij = torch::sin(thetas[thetaIndx]).item();

			// Compute row updates
			newRowUi.zero_();
			newRowUj.zero_();

			// newRowUi = U[i,:]*cij - U[j,:]*sij;
			newRowUi.add_(U.index({ i,Slice() }), cij);
			newRowUi.add_(U.index({ j,Slice() }), -sij);

			// newRowUj = U[i,:]*sij + U[j,:]*cij;
			newRowUj.add_(U.index({ i,Slice() }), sij);
			newRowUj.add_(U.index({ j,Slice() }), cij);

			// U[i,:] = newRowUi;
			U.index_put_({ i, Slice() }, newRowUi);
			// U[j,:] = newRowUj;
			U.index_put_({ j, Slice() }, newRowUj);
		}
	}
	return U;
}


torch::Tensor rotMatBck(torch::Tensor thetas, torch::Tensor U, torch::Tensor G)
{
	auto N = U.size(0);
	auto rotMatConstants = determineRotMatConstants(thetas.size(0), N);
	auto dMax = std::get<0>(rotMatConstants);
	auto deadVar = std::get<1>(rotMatConstants);
	auto Ntilde = std::get<2>(rotMatConstants);

	// In rotMatFwd, U is given these same properties
	auto tensOptions = torch::TensorOptions().dtype(thetas.dtype()).device(thetas.device());
	auto JVP = torch::zeros_like(thetas, tensOptions);

	auto A = torch::zeros({ 1,N }, tensOptions);
	
	auto M = torch::zeros({ N,N }, tensOptions);
	M.index_put_({ Slice(0,G.size(1)), Slice() }, G.detach().t());

	auto Uf = torch::clone(U);

	auto C = torch::cos(thetas);
	auto S = torch::sin(thetas);

	// Temp row/col variables
	auto newVi = torch::zeros({ N }, tensOptions);
	auto newVj = torch::zeros({ N }, tensOptions);

	for (int64_t step = Ntilde - 2; step >= 0; step--)
	{
		for (int64_t blockIndx = 0; blockIndx < Ntilde / 2; blockIndx++)
		{
			auto rowIndices = determineRowIndexPair(blockIndx, Ntilde, step);
			int i = rowIndices.first;
			int j = rowIndices.second;

			if (areRowIndicesOutOfRange(i, j, deadVar, dMax))
			{
				continue;
			}

			// Rel to N
			int64_t thetaIndx = i * N - (i + 2) * (i + 1) / 2 + j;
			auto cij = C[thetaIndx].item();
			auto sij = S[thetaIndx].item();

			// Compute row updates
			{
				newVi.zero_();
				newVj.zero_();

				// newVi = M[i,:]*cij - M[j,:]*sij;
				newVi.add_(M.index({i, Slice()}), cij);
				newVi.add_(M.index({j, Slice()}), -sij);

				// newVj = M[i,:]*sij + M[j,:]*cij;
				newVj.add_(M.index({i,Slice()}), sij);
				newVj.add_(M.index({j, Slice()}), cij);

				// M[i,:] = newVi;
				M.index_put_({i, Slice()}, newVi);
				// M[j,:] = newVj;
				M.index_put_({j, Slice()}, newVj);
			}

			// Compute col updates
			{
				newVi.zero_();
				newVj.zero_();

				// newVi = Uf[:,i]*cij - Uf[:,j]*sij;
				newVi.add_(Uf.index({Slice(), i}), cij);
				newVi.add_(Uf.index({Slice(), j}), -sij);

				// newVj = Uf[:,i]*sij + Uf[:,j]*cij;
				newVj.add_(Uf.index({Slice(), i}), sij);
				newVj.add_(Uf.index({Slice(), j}), cij);

				// Uf[:,i] = newVi;
				Uf.index_put_({Slice(), i}, newVi);
				// Uf[:,j] = newVj;
				Uf.index_put_({Slice(), j}, newVj);
			}

			A.index_put_({0, Slice()}, M.index({i, Slice()}).mul(Uf.index({Slice(), j})));
			A.index({0, Slice()}).add_(-M.index({j, Slice()}).mul(Uf.index({Slice(), i})));

			JVP[thetaIndx] = A.index( 0, Slice()}).sum();
		}
	}
	return JVP;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &rotMatFwd, "Rotation Matrix forward");
	m.def("backward", &rotMatBck, "Rotation Matrix backward");
}
