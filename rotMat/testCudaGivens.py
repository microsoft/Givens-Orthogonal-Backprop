# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import cuda.GivensRotations as GivensRotations
import time
import numpy as np
import rotMatcuda
from groundTruthGivens import sequentialGivens

# This script is not meant as a rigorous benchmark; only to verify correctness against naive autodiff

device = torch.device('cuda')
dtype = torch.float32

# To have same results
torch.manual_seed(10)
dispResults = False

# SEQUENTIAL WILL TAKE FOREVER IF TEAM SIZE IS NOT SMALL, MAKE SURE TO CHANGE IT BEFORE TESTING
Ns = [6,15,33] #[15, 33, 30]

#teamSize = rotMatcuda.getTeamSize()
#if teamSize <= 16: Ns = [teamSize, teamSize+1, teamSize *2 + 1]
#if teamSize <= 8: Ns.append(teamSize*4 + int(teamSize/3))

Ms = [min(Ns), 5, 1]
batch_sizes = [15, 3 ,33, 65]

parameters = [[x] for x in Ns]
parameters = [x + [m] for m in Ms for x in parameters]
parameters = [x + [b] for b in batch_sizes for x in parameters]
parameters = [x + [isId] for isId in [True, False] for x in parameters]
parameters = [x + [rr] for rr in [False, False] for x in parameters]


for N, M, batch, XisId, isTeamRR in parameters: 
    K = N-M
    if isTeamRR:
        rotUnit = GivensRotations.RotMatOpt(N, M).to(device)
    else:
        rotUnit = GivensRotations.RotMat(N, M).to(device)
        
    rotUnit.thetas.data = torch.ones(rotUnit.thetas.data.size()).to(device) * np.pi * 2
    thetas = rotUnit.thetas.detach().to(torch.device('cpu')).requires_grad_(True)

    G = torch.randn(N,batch).to(dtype).to(device)

    if XisId:
        X = torch.eye(N, batch).to(dtype).to(device)
    else:
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

    # Milliseconds
    forwardMilliseconds = startFwd.elapsed_time(endFwd)
    backwardMilliseconds = startBck.elapsed_time(endBck)
    if dispResults:
        print( '(In ms) Forward time: {0:.3f} | Backward time: {1:.3f}'.format(forwardMilliseconds,backwardMilliseconds) ) # milliseconds
    gradCustom = rotUnit.thetas.grad.to(torch.device('cpu'))
    Ucustom = Ucustom.to(torch.device('cpu'))

    if dispResults:
        print("Custom result:")
        print("U:")
        print(Ucustom)
        print("thetaGrad:")
        print(gradCustom)

    tstart = time.time()
    UPyTorch = sequentialGivens(UPyTorch, thetas, M)

    G = G.to(torch.device('cpu'))
    loss = (G*UPyTorch).sum()
    loss.backward()
    if dispResults:
        print(" (in seconds) torch time: "+str(time.time()-tstart))

    if dispResults:
        print("Torch autodiff result:")
        print("U:")
        print(UPyTorch)
        print("thetaGrad:")
        print(thetas.grad)

    if dispResults:
        print("\n\nComparison of custom and autodiff:\n---------------------------------")
        print("Max abs deviation of forwards: ")
        print(torch.absolute(Ucustom-UPyTorch).max())
        print("Max abs deviation of grads: ")
        print(torch.absolute(thetas.grad-gradCustom).max())

    msgInfo = "N: " + str(N) + " M: " + str(M) + " batch: " + str(batch) + " XisId: " + str(XisId) + " usingTeamRR: " + str(isTeamRR)
    if  N == batch and XisId:
        if dispResults: print(torch.round(Ucustom @ Ucustom.t()), "\nDETERMINANT:\n",torch.det(Ucustom))
        torch.testing.assert_close(Ucustom @ Ucustom.t(), torch.eye(N, N), msg=msgInfo)
    
    if not isTeamRR:
        torch.testing.assert_close(Ucustom, UPyTorch, check_layout=True, msg= msgInfo)
        torch.testing.assert_close(thetas.grad, gradCustom, check_layout=True, msg=msgInfo)
    torch.cuda.synchronize()

print("All tests passed!")
