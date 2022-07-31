import torch
import time
import numpy as np

def sequentialGivens(UPyTorch, thetas, M):
    N = UPyTorch.size(0)
    K=N-M
    
    dMax = N-1
    if M < N-1:
        dMax -= K

    Ntilde = N
    deadNode = -1
    if N%2 == 1:
        Ntilde += 1
        deadNode = Ntilde-1

    for step in range(Ntilde-2,-1,-1):
        for blockIndx in range(int(Ntilde/2)):

            if blockIndx == 0:
                i = 0
            else:
                i = (blockIndx-1+step)%(Ntilde-1)+1
            j = ( (Ntilde-1)-blockIndx-1+step )%(Ntilde-1) + 1
            if i > j:
                temp = i
                i = j
                j = temp

            if (i > dMax and j > dMax) or j==deadNode:
                continue

            thetaIndx = int( i*N - (i+2)*(i+1)/2 + j )
            cij = torch.cos(thetas[thetaIndx])
            sij = torch.sin(thetas[thetaIndx])

            GivMat = torch.eye(N)
            GivMat[i,i] = cij
            GivMat[j,j] = cij
            GivMat[i,j] = -sij
            GivMat[j,i] = sij
            UPyTorch = GivMat.matmul(UPyTorch)
    return UPyTorch