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
dispResults = True

# SEQUENTIAL WILL TAKE FOREVER IF TEAM SIZE IS NOT SMALL, MAKE SURE TO CHANGE IT BEFORE TESTING
Ns = [6, 32, 15, 30, 33]

teamSize = rotMatcuda.getTeamSize()
if teamSize <= 16: Ns = [teamSize, teamSize+1, teamSize *2 + 1]
if teamSize <= 8: Ns.append(teamSize*4 + int(teamSize/3))

Ms = [min(Ns)]#, 5, 1]
batch_sizes = Ns#[6, 15, 3 ,33, 65]

parameters = [[x] for x in Ns]
parameters = [x + [m] for m in Ms for x in parameters]
parameters = [x + [b] for b in batch_sizes for x in parameters]
parameters = [x + [isId] for isId in [True, False] for x in parameters]
parameters = [x + [rr] for rr in [True] for x in parameters]

for N, M, batch, XisId, isTeamRR in parameters: 
    K = N-M
    if isTeamRR:
        rotUnit = GivensRotations.RotMatOpt(N, M).to(device)
    else:
        rotUnit = GivensRotations.RotMat(N, M).to(device)
        
    rotUnit.thetas.data = torch.randn(rotUnit.thetas.data.size()).to(device).requires_grad_(True)
    thetas = torch.clone(rotUnit.thetas).detach().to(torch.device('cpu'))
    thetas.requires_grad_(True)

    G = torch.randn(N,batch).to(dtype).to(device)

    if XisId:
        X = torch.eye(N, batch).to(dtype).to(device).requires_grad_(True)
    else:
        X = torch.randn(N, batch).to(dtype).to(device).requires_grad_(True)
    
    UPyTorch = torch.clone(X).to('cpu').requires_grad_(True)

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

    playerCountInTeam = teamSize if isTeamRR else -1
    UPyTorch = sequentialGivens(UPyTorch, thetas, M, playerCountInTeam)

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

    print("TEAM SIZE:", teamSize)
    print()
    msgInfo = "N: " + str(N) + " M: " + str(M) + " batch: " + str(batch) + " XisId: " + str(XisId) + " usingTeamRR: " + str(isTeamRR)
    if  N == batch and XisId:
        if dispResults: 
            print("UPyTorch\n",torch.round(UPyTorch @ UPyTorch.t()), "\nDETERMINANT:\n",torch.det(UPyTorch))
            print("UCUSTOM\n",torch.round(Ucustom @ Ucustom.t()), "\nDETERMINANT:\n",torch.det(Ucustom))
        torch.testing.assert_close(Ucustom @ Ucustom.t(), torch.eye(N, N), msg=msgInfo)
    
    print("isclose: ", torch.isclose(Ucustom, UPyTorch))
    print(Ucustom[-1], "\n", UPyTorch[-1])
    torch.testing.assert_close(Ucustom, UPyTorch, check_layout=True, msg= msgInfo)
    #torch.testing.assert_close(thetas.grad, gradCustom, check_layout=True, msg=msgInfo)
    torch.cuda.synchronize()

print("All tests passed!")
