import torch
import time
import numpy as np
from math import ceil

def sequentialGivens(UPyTorch, thetas, M, isTeamRR, playerCountInTeam):
    if isTeamRR:
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
    x = 0
    N = UPyTorch.size(0)
    K=N-M
    
    dMax = N-1
    if M < N-1:
        dMax -= K

    Ntilde = N
    deadNode = Ntilde
    if N%2 == 1:
        Ntilde += 1
    
    teamCount = int(Ntilde// teamSize) + int(Ntilde % teamSize != 0)
    
    for teamNo in range(0, teamCount):
        start = teamNo*teamSize
        currentTeamSize = teamSize
        if start + currentTeamSize > Ntilde:
            currentTeamSize = Ntilde - start

        for step in range(0, currentTeamSize-1):
            for blockIndx in range(int(currentTeamSize/2)-1,-1,-1):
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

                #print("torch: ", i, j, "-> ", "S", round(sij.item(),6),"C: ",round(cij.item(),6))
                x+=1


    # Add dummy team Index for team tournament
    i = j = None

    dummyTeamIndex = teamCount
    if(teamCount % 2 != 0):
        teamCount += 1
    blocksCount = int(Ntilde/2/teamSize) + int(Ntilde/2 % teamSize != 0)


    for matchUpRound in range(teamCount-2,-1,-1):
        for blockIndx in range(blocksCount):

            team_i, team_j = getRowIndexPair(blockIndx, teamCount, matchUpRound)
            if team_j == dummyTeamIndex : 
                continue
            

            iStart = teamSize * team_i
            jStart = teamSize * team_j
            for teamMatchStep in range(teamSize):

                for step in range(teamSize):
                    i = iStart + step

                    j = jStart + step + teamMatchStep
                    if j >= jStart + teamSize:
                        j -= teamSize

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
                    
                    #print("torch: ", i, j, "->", "S", round(sij.item(),6), "C: ", round(cij.item(),6))
                    x += 1
    
    #print("theta count: ", x)

    if x == 0:
        thetas.grad = torch.zeros(thetas.size(0)) if not thetas.grad else thetas.grad
    return UPyTorch
