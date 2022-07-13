import torch
import rotMatcpp
import time

# This script is not meant as a rigorous benchmark; only to verify correctness against naive autodiff

# To have same results
torch.manual_seed(0)

N = 10
M = 3
K = N-M
useFrame = True

nThetas = int(N*(N-1)/2)
# To drop angle params we need to have at least the last *pair* of dimensions left out
if K > 1:
    nThetas -= int(K*(K-1)/2)
print("nThetas=" +str(nThetas) )
dispResults = True

if useFrame is True:
    nCols = M
else:
    nCols = N


# Loss grad when the loss is weighted sum of the els of U
G = torch.randn(N,nCols)
thetas = torch.randn(nThetas)

tstart = time.time()
U = rotMatcpp.forward(thetas,N)
JVP = rotMatcpp.backward(thetas,U,G)
print("cpp time: "+str(time.time()-tstart))

if dispResults:
    print("rotMatcpp result:")
    print("U:")
    print(U)
    print("thetaGrad:")
    print(JVP)

thetas.requires_grad_(True)
Uauto = torch.eye(N,nCols)

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
        Uauto = GivMat.mm(Uauto)

        # Torch autodiff does not like this!
        #newRowi = U[i,:]*cij - U[j,:]*sij
        #newRowj = U[i,:]*sij + U[j,:]*cij
        #U[i,:] = newRowi
        #U[j,:] = newRowj

loss = (G*Uauto).sum()
loss.backward()
print("torch time: "+str(time.time()-tstart))

if dispResults:
    print("Torch autodiff result:")
    print("U:")
    print(Uauto)
    print("thetaGrad:")
    print(thetas.grad)
    # print(E)

if dispResults:
    print("\n\nComparison of custom and autodiff:\n---------------------------------")
    print("Max abs deviation of forwards: ")
    print(torch.absolute(U[:,0:nCols]-Uauto).max())
    print("Max abs deviation of grads: ")
    print(torch.absolute(thetas.grad-JVP).max())


