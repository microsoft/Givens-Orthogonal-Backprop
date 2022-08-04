# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import cuda.GivensRotations as GivensRotations
import time
import numpy as np
import rotMatcuda

# This script is not meant as a rigorous benchmark; only to verify correctness against naive autodiff

device = torch.device('cuda')
dtype = torch.float32

# To have same results
torch.manual_seed(10)
dispResults = False

Ns = [3, 32, 6, 15, 33]

teamSize = rotMatcuda.getTeamSize()
print("TEAM SIZE:", teamSize, "\n")
if teamSize <= 16: Ns = [teamSize, teamSize+1, teamSize *2 + 1]
if teamSize <= 8: Ns.append(teamSize*4 + int(teamSize/3))

Ms =Ns
batch_sizes = [6, 15, 3,17, 25,26, 27,33]

parameters = [[x] for x in Ns]
parameters = [x + [m] for m in Ms for x in parameters if  m <= x[0]]
parameters = [x + [b] for b in batch_sizes for x in parameters]
parameters = [x + [isId] for isId in [True, False] for x in parameters]
parameters = [x + [rr] for rr in [False, True] for x in parameters]

for index, (N, M, batch, XisId, isTeamRR) in enumerate(parameters,start=1): 
    
    if isTeamRR:
        rotUnit = GivensRotations.RotMatOpt(N, M).to(device)
    else:
        rotUnit = GivensRotations.RotMat(N, M).to(device)
    
    rotUnit = GivensRotations.RotMat(N, M).to(device)
    rotUnit.thetas.data = torch.ones(rotUnit.thetas.data.size()).to(device) * np.pi * 2
    G = torch.randn(N,batch).to(dtype).to(device)

    if XisId:
        X = torch.autograd.Variable(torch.eye(N, batch).to(dtype).to(device), requires_grad=True)
    else:
        X = torch.autograd.Variable(torch.randn(N, batch).to(dtype).to(device), requires_grad=True)
    
    XExplicit = torch.autograd.Variable(torch.clone(X), requires_grad=True)
    GExplicit = torch.clone(G)

    # START WITH IMPLICIT U CALCULATION
    startFwd = torch.cuda.Event(enable_timing=True)
    endFwd = torch.cuda.Event(enable_timing=True)
    startFwd.record()
    UXcustom = rotUnit.forward(X)
    endFwd.record()
    
    loss = torch.sum(G*UXcustom)
    startBck = torch.cuda.Event(enable_timing=True)
    endBck = torch.cuda.Event(enable_timing=True)
    startBck.record()
    loss.backward()
    endBck.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    # Milliseconds
    forwardMilliseconds = startFwd.elapsed_time(endFwd)
    backwardMilliseconds = startBck.elapsed_time(endBck)
    if dispResults:
        print( '(In ms) Forward time: {0:.3f} | Backward time: {1:.3f}'.format(forwardMilliseconds,backwardMilliseconds) ) # milliseconds

    XgradCustom = X.grad.to(torch.device('cpu'))
    UXcustom = UXcustom.to(torch.device('cpu'))

    if dispResults:
        print("Custom result:")
        print("UX:")
        print(UXcustom)
        print("XGrad:")
        print(XgradCustom)

    # NOW FOR EXPLICIT U CALC
    rotUnit.zero_grad()
    startFwd = torch.cuda.Event(enable_timing=True)
    endFwd = torch.cuda.Event(enable_timing=True)
    startFwd.record()
    UXExplicit = rotUnit.getU() @ XExplicit
    endFwd.record()
    torch.cuda.synchronize()

    loss = torch.sum(GExplicit*UXExplicit)
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
    XExplicitgradCustom = XExplicit.grad.to(torch.device('cpu'))
    UXExplicit = UXExplicit.to(torch.device('cpu'))

    if dispResults:
        print("Explicit Custom result:")
        print("UX Explicit:")
        print(UXExplicit)
        print("XGrad Explicit:")
        print(XExplicitgradCustom)

    if dispResults:
        print("\n\nComparison of custom and autodiff:\n---------------------------------")
        print("Max abs deviation of forwards: ")
        print(torch.absolute(UXcustom-UXExplicit).max())
        print("Max abs deviation of grads: ")
        print(torch.absolute(XgradCustom-XExplicitgradCustom).max())

    msgInfo = "N: " + str(N) + " M: " + str(M) + " batch: " + str(batch) + " XisId: " + str(XisId)
    torch.testing.assert_close(UXcustom, UXExplicit, check_layout=True, msg= msgInfo)
    torch.testing.assert_close(XgradCustom, XExplicitgradCustom, check_layout=True, msg=msgInfo)
    if N == batch and XisId:
        torch.testing.assert_close(UXcustom @ UXcustom.t(), torch.eye(N,N), msg=msgInfo)
        torch.testing.assert_close(UXExplicit @ UXExplicit.t(), torch.eye(N,N), msg=msgInfo)
    torch.cuda.synchronize()
    print(index, "Test passed!")

print("All tests passed!")



