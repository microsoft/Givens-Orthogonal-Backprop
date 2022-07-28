# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import cuda.GivensRotations as GivensRotations
import time
import numpy as np

# This script is not meant as a rigorous benchmark; only to verify correctness against naive autodiff

device = torch.device('cuda')
dtype = torch.float32

# To have same results
torch.manual_seed(0)

dispResults = True

N = 15
M = 15
K = N-M

rotUnit = GivensRotations.RotMat(N, N).to(device)
torch.nn.init.uniform_(rotUnit.thetas, 0, 4*np.pi )

batch = 15
# Loss grad when the loss is weighted sum of the els of U
G = torch.randn(N,batch).to(dtype).to(device)
X = torch.eye(N, batch).to(dtype).to(device)

# You *cannot* use time.time() to time cuda-enabled functions! The cpu
# proceeds asynchronously, leading to ludicrous underestimates.
#
# The elapsed_time function returns time in *milliseconds*

startFwd = torch.cuda.Event(enable_timing=True)
endFwd = torch.cuda.Event(enable_timing=True)
startFwd.record()
Ucustom = rotUnit.forward(X)
endFwd.record()
# Waits for everything to finish running
torch.cuda.synchronize()

#print(G.size(), Ucustom.size())
loss = torch.sum(G*Ucustom)

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

print( 'Forward time: {0:.3f} | Backward time: {1:.3f}'.format(forwardMilliseconds,backwardMilliseconds) ) # milliseconds

gradCustom = rotUnit.thetas.grad.to(torch.device('cpu'))
Ucustom = Ucustom.to(torch.device('cpu'))
G = G.to(torch.device('cpu'))

if dispResults:
    print("Custom result:")
    print("U:")
    print(Ucustom)
    print("thetaGrad:")
    print(gradCustom)

# Compare to direct with PyTorch autodiff

thetas = rotUnit.thetas.detach().to(torch.device('cpu')).requires_grad_(True)
UPyTorch = torch.clone(X).to('cpu')

# (i,j) with indices larger than dMax are left out
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
        UPyTorch = GivMat.mm(UPyTorch)

        # Torch autodiff does not like this!
        #newRowi = U[i,:]*cij - U[j,:]*sij
        #newRowj = U[i,:]*sij + U[j,:]*cij
        #U[i,:] = newRowi
        #U[j,:] = newRowj

loss = (G*UPyTorch).sum()
loss.backward()
print("torch time: "+str(time.time()-tstart))

if dispResults:
    print("Torch autodiff result:")
    print("U:")
    print(UPyTorch)
    print("thetaGrad:")
    print(thetas.grad)
    # print(E)

if dispResults:
    print("\n\nComparison of custom and autodiff:\n---------------------------------")
    print("Max abs deviation of forwards: ")
    print(torch.absolute(Ucustom-UPyTorch).max())
    print("Max abs deviation of grads: ")
    print(torch.absolute(thetas.grad-gradCustom).max())
    #print(torch.isclose(thetas.grad, gradCustom))
