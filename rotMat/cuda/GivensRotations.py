# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import rotMatcuda
import numpy as np

def getThetaCount(N, M):
    K = N-M
    return int(N*(N-1)/2) if K <= 1 else int(N*(N-1)/2) - int(K*(K-1)/2)
    
class RotMatFunction(torch.autograd.Function):
    '''
        If the inputs are expected to be re-used, all cloning must be done here!
    '''
    @staticmethod
    def forward(ctx, x, thetas):
        ux = rotMatcuda.forward(torch.clone(x.detach()), thetas.detach())
        ctx.save_for_backward(thetas, ux)
        return  ux

    @staticmethod
    def backward(ctx, lossGrad):
        thetas, ux = ctx.saved_tensors
        lossGrad, thetaGrad = rotMatcuda.backward(thetas.detach(), torch.clone(ux).detach(), lossGrad.detach().contiguous())
        return lossGrad, thetaGrad

class RotMat(torch.nn.Module):
    def __init__(self, N, M=None):
        super(RotMat, self).__init__()

        M = N if M is None else M # default is fully parametrized
        if M > N:
            raise Exception("M must be <= N")
        
        self.thetas = torch.nn.Parameter(torch.zeros(getThetaCount(N,M)))
        self.N, self.M, self.U = N, M, None


    def forward(self, X):
        return RotMatFunction.apply(X, self.thetas)

    def getU(self, forward_pass=True):
        if not forward_pass:
            if not self.U:
                identity = torch.eye(self.N, self.N).to(self.thetas.get_device()).to(self.thetas.dtype)
                self.U = RotMatFunction.apply(identity, self.thetas)
            return self.U

        identity = torch.eye(self.N, self.N).to(self.thetas.get_device()).to(self.thetas.dtype)
        return RotMatFunction.apply(identity, self.thetas)


'''
********************************************
    SAME ROUND ROBIN GIVENS ROTATIONS 
    WITH OPTIMIZED FORWARD/BACKPROP
    BELOW
********************************************
'''

class RotMatOptFunction(torch.autograd.Function):
    '''
        If the inputs are expected to be re-used, all cloning must be done here!
    '''

    @staticmethod
    def forward(ctx, x, thetas):
        ux = rotMatcuda.forwardTeamRR(torch.clone(x.detach()), thetas.detach())
        ctx.save_for_backward(thetas, ux)
        return  ux

    @staticmethod
    def backward(ctx, lossGrad):
        thetas, ux = ctx.saved_tensors
        lossGrad, thetaGrad = rotMatcuda.backwardTeamRR(thetas.detach(), torch.clone(ux).detach(), lossGrad.detach().contiguous())
        return lossGrad, thetaGrad

class RotMatOpt(RotMat):
    def __init__(self, N, M=None):
        RotMat.__init__(self, N,M)
    
    def forward(self, X):
        return RotMatOptFunction.apply(X, self.thetas)

    def getU(self, forward_pass=True):
        if not forward_pass:
            if not self.U:
                identity = torch.eye(self.N, self.N).to(self.thetas.get_device()).to(self.thetas.dtype)
                self.U = RotMatOptFunction.apply(identity, self.thetas)
            return self.U

        identity = torch.eye(self.N, self.N).to(self.thetas.get_device()).to(self.thetas.dtype)
        return RotMatOptFunction.apply(identity, self.thetas)
