import torch
import rotMatcpp
import math

class rotMatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thetas, N, nCols ):
        
        U = rotMatcpp.forward(thetas, N)
        ctx.save_for_backward(thetas, U)

        return U[:,0:nCols]

    @staticmethod
    def backward(ctx, lossGrad):

        thetas, U = ctx.saved_tensors
        thetaGrad = rotMatcpp.backward(thetas, U, lossGrad)

        # "forward" took 3 inputs but we don't want the grad wrt N or nCols!
        return thetaGrad, None, None

    
class RotMat(torch.nn.Module):

    # Note that rotMat doesn't take any "inputs"; its behavior is
    # given solely by thetas.

    # If N is 1, simply returns the unit tensor
    
    def __init__(self, N, M=None, useFrame=False):
        super(RotMat, self).__init__()
        if M is None:
            M = N
        if M > N:
            raise Exception("M must be <= N")
        self.N = N
        self.M = M
        K = N-M
        nThetas = int(N*(N-1)/2)
        if K > 1:
            nThetas -= int(K*(K-1)/2)
        if useFrame is True:
            self.nCols = M
        else:
            self.nCols = N
        # Uniform angles- not the Haar measure! Latter requires
        # e.g. sampling appropriate univariate sinusoid powers
        self.thetas = torch.nn.Parameter( torch.empty(nThetas) )
        torch.nn.init.uniform_(self.thetas, 0, 2*math.pi )
            
    def forward(self):
        return rotMatFunction.apply( self.thetas, self.N, self.nCols )
