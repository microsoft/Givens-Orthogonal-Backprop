import torch
import time
import numpy as np
from math import ceil

def sequentialGivens(UPyTorch, thetas, M, playerCountInTeam):
    if playerCountInTeam > 0:
        return sequentialGivensTeamRR(UPyTorch, thetas, M, playerCountInTeam)
    return sequentialGivensCircleMethod(UPyTorch, thetas, M)


def getRowIndexPair(indexx, allRows, s):
    if indexx == 0:
        i = 0
    else:
        i = (indexx-1+s)%(allRows-1)+1
    j = ( (allRows-1)-indexx-1+s )%(allRows-1) + 1
    
    if (i < j):
        return i, j 
    return j, i


def sequentialGivensCircleMethod(UPyTorch, thetas, M):
    N = UPyTorch.size(0)
    K=N-M
    
    dMax = N-1
    if M < N-1:
        dMax -= K

    Ntilde = N
    deadNode = Ntilde
    if N%2 == 1:
        Ntilde += 1
        deadNode = Ntilde-1

    for step in range(Ntilde-2,-1,-1):
        for blockIndx in range(int(Ntilde/2)):

            i, j = getRowIndexPair(blockIndx, Ntilde, step)
            if (i > dMax and j > dMax) or j>=deadNode:
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

def sequentialGivensTeamRR(UPyTorch, thetas, M, teamSize):
    N = UPyTorch.size(0)
    K=N-M
    
    dMax = N-1
    if M < N-1:
        dMax -= K

    Ntilde = N
    deadNode = Ntilde
    if N%2 == 1:
        Ntilde += 1
    
    teamcount = int(Ntilde/ teamSize) + int(Ntilde % teamSize != 0)
    for teamNo in range(0, teamcount):
        start = teamNo*teamSize
        currentTeamSize = teamSize
        if start + currentTeamSize > Ntilde:
            currentTeamSize = Ntilde - start
        
        for blockIndx in range(int(currentTeamSize/2)):
            print()
            for step in range(0, currentTeamSize-1):
                i, j = getRowIndexPair(blockIndx, currentTeamSize, step)
                i += start
                j += start
                #print("blockIndx", blockIndx, "i, j", i,j)

                if (i > dMax and j > dMax) or j>=deadNode:
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