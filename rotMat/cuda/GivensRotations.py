# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import rotMatcuda
import numpy as np

class RotMatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thetas):
        ux = rotMatcuda.forward(torch.clone(x), thetas)
        ctx.save_for_backward(thetas, ux)
        return  ux

    @staticmethod
    def backward(ctx, lossGrad):
        thetas, ux = ctx.saved_tensors
        thetaGrad = rotMatcuda.backward(thetas, torch.clone(ux).detach(), lossGrad.contiguous().detach())
        return lossGrad, thetaGrad


class RotMat(torch.nn.Module):
    def __init__(self, N, M=None):
        super(RotMat, self).__init__()

        M = N if M is None else M
        if M > N:
            raise Exception("M must be <= N")
        
        K = N-M
        nThetas = int(N*(N-1)/2) if K <= 1 else int(N*(N-1)/2) - int(K*(K-1)/2)
        self.thetas = torch.nn.Parameter(torch.zeros(nThetas))

        self.N = N
        self.M = M
        self.U = None

    def forward(self, X):
        return RotMatFunction.apply(X, self.thetas)

    def explicit_forward(self, X):
        return getU() @ X

    def getU(self, forward_pass=True):
        if not forward_pass:
            if not self.U:
                identity = torch.eye(self.N, self.M).to(self.thetas.get_device()).to(self.thetas.dtype)
                self.U = RotMatFunction.apply(identity, self.thetas)
            return self.U

        identity = torch.eye(self.N, self.M).to(self.thetas.get_device()).to(self.thetas.dtype)
        return RotMatFunction.apply(identity, self.thetas)


