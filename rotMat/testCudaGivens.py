# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import cuda.GivensRotations as GivensRotations
import time
import numpy as np

# This script is not meant as a rigorous benchmark;
# only to verify correctness against naive autodiff

device = torch.device('cuda')
dtype = torch.float32

# To have same results
torch.manual_seed(10)
dispResults = False

parameters = [
    [15, 15, 15],
    [15, 15, 3],
    [15, 15, 32],
    [15, 5, 15],
    [15, 5, 3],
    [15, 5, 32],
    [33, 15, 15],
    [33, 15, 3],
    [33, 15, 32],
    [33, 5, 15],
    [33, 5, 3],
    [33, 5, 32],
    [33, 5, 33]
    ]
parameters = [[x] + param for x in [True,False] for param in parameters]


for XisId, N, M, batch in parameters: 
    K = N-M
    rotUnit = GivensRotations.RotMat(N, M).to(device)
    rotUnit.thetas.data = torch.ones(rotUnit.thetas.data.size()).to(device) * np.pi * 2
    thetas = rotUnit.thetas.detach().to(torch.device('cpu')).requires_grad_(True)

    G = torch.randn(N,batch).to(dtype).to(device)
    if XisId:
        X = torch.eye(N, batch).to(dtype).to(device)
    if not XisId:
        X = torch.randn(N, batch).to(dtype).to(device)

    UPyTorch = torch.clone(X).to('cpu')

    startFwd = torch.cuda.Event(enable_timing=True)
    endFwd = torch.cuda.Event(enable_timing=True)
    startFwd.record()
    Ucustom = rotUnit.forward(X)
    endFwd.record()
    torch.cuda.synchronize()

    loss = torch.sum(G*Ucustom)
    startBck = torch.cuda.Event(enable_timing=True)
    endBck = torch.cuda.Event(enable_timing=True)
    startBck.record()
    loss.backward()
    endBck.record()
    torch.cuda.synchronize()

    forwardMilliseconds = startFwd.elapsed_time(endFwd)
    backwardMilliseconds = startBck.elapsed_time(endBck)
    if dispResults:
        print( '(In ms) Forward time: {0:.3f} | Backward time: {1:.3f}'.format(forwardMilliseconds,backwardMilliseconds) ) # milliseconds
    gradCustom = rotUnit.thetas.grad.to(torch.device('cpu'))
    Ucustom = Ucustom.to(torch.device('cpu'))

    if dispResults:
        print("Custom result:\nU:\n", Ucustom, "\nthetaGrad:\n", gradCustom)

    dMax = N-1
    if M < N-1:
        dMax -= K
    Ntilde = N
    deadNode = -1
    if N%2 == 1:
        Ntilde += 1
        deadNode = Ntilde-1

    tstart = time.time()
    E = list()
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

            E.append( (i,j) )
            thetaIndx = int( i*N - (i+2)*(i+1)/2 + j )
            cij = torch.cos(thetas[thetaIndx])
            sij = torch.sin(thetas[thetaIndx])

            GivMat = torch.eye(N)
            GivMat[i,i] = cij
            GivMat[j,j] = cij
            GivMat[i,j] = -sij
            GivMat[j,i] = sij
            UPyTorch = GivMat.matmul(UPyTorch)

    G = G.to(torch.device('cpu'))
    loss = (G*UPyTorch).sum()
    loss.backward()

    if dispResults:
        print(" (in seconds) torch time: "+str(time.time()-tstart))
        print("Torch autodiff result:\nU:\n", UPyTorch, "thetaGrad:\n", thetas.grad)
        print("\n\nComparison of custom and autodiff:\n---------------------------------\n")
        print("Max abs deviation of forwards:\n", torch.absolute(Ucustom-UPyTorch).max())
        print("Max abs deviation of grads: \n", torch.absolute(thetas.grad-gradCustom).max())

    msgInfo = "N: " + str(N) + " M: " + str(M) + " batch: " + str(batch) + " XisId: " + str(XisId)
    torch.testing.assert_close(Ucustom, UPyTorch, check_layout=True, msg= msgInfo)
    torch.testing.assert_close(thetas.grad, gradCustom, check_layout=True, msg=msgInfo)
    if N == batch and XisId:
        torch.testing.assert_close(Ucustom @ Ucustom.t(), torch.eye(N,N), msg=msgInfo)
    torch.cuda.synchronize()

print("All tests passed!")
