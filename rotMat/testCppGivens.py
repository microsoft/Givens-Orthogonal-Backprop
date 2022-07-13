import torch
import cpp.GivensRotations as GivensRotations
import time

# This script is not meant as a rigorous benchmark; only to verify correctness against naive autodiff

# To have same results
torch.manual_seed(0)

dispResults = True

N = 10
M = 3
K = N-M
useFrame = True
if useFrame is True:
    nCols = M
else:
    nCols = N



rotUnit = GivensRotations.RotMat(N,M,useFrame)
tstart = time.time()
Ucustom = rotUnit.forward()

# Loss grad when the loss is weighted sum of the els of U
G = torch.randn(N,nCols)
loss = torch.sum(G*Ucustom)

loss.backward()
print("cpp time: "+str(time.time()-tstart))
gradCustom = rotUnit.thetas.grad.clone()


if dispResults:
    print("Custom result:")
    print("U:")
    print(Ucustom)
    print("thetaGrad:")
    print(gradCustom)

# Compare to direct with PyTorch autodiff

thetas = rotUnit.thetas.detach().clone().requires_grad_(True)
UPyTorch = torch.eye(N,nCols)

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
print(thetas.dtype)
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
