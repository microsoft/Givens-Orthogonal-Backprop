# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import rotMatcuda
import math

class RotMatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thetas, N, nCols, threadsperblock):

        U = rotMatcuda.forward(thetas, N, threadsperblock)
        ctx.save_for_backward(thetas, U)
        ctx.in1 = threadsperblock

        if N == nCols:
            return U
        return U[:,0:nCols]

    @staticmethod
    def backward(ctx, lossGrad):

        thetas, U = ctx.saved_tensors
        threadsperblock = ctx.in1
        # Force the grad wrt outputs to be contiguous
        thetaGrad = rotMatcuda.backward( thetas, U, lossGrad.contiguous(),threadsperblock )

        # "forward" took 3 inputs but we don't want the grad wrt N or nCols!
        return thetaGrad, None, None, None


class RotMat(torch.nn.Module):

    # Note that RotMat doesn't take any "inputs"; its behavior is
    # given solely by thetas.

    # If N is 1, simply returns the unit tensor

    def __init__(self, N, M=None, useFrame=False, threadsperblock=512):
        super(RotMat, self).__init__()

        M = N if M is None else M
        if M > N:
            raise Exception("M must be <= N")

        self.N = N
        self.M = M
        K = N-M
        nThetas = int(N*(N-1)/2) if K <= 1 else int(N*(N-1)/2) - int(K*(K-1)/2)
        
        self.thetas = torch.nn.Parameter( torch.empty(nThetas) )
        torch.nn.init.uniform_(self.thetas, 0, 4*math.pi )

        self.nCols = M if useFrame else N
        self.U = None

        # To benchmark without having to rebuild
        self.threadsperblock = threadsperblock

    def forward(self):
        return self.get_orthogonal_matrix()

    def get_orthogonal_matrix(self, forward_pass=True):
        if forward_pass or not self.U:
            self.U = RotMatFunction.apply(self.thetas, self.N, self.nCols, self.threadsperblock)
        return self.U


