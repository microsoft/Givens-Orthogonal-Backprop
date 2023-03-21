"""
    Produces small version of Figure 3 from https://arxiv.org/pdf/2009.13977

    Differences.

    - To allow fast experiment we have repeats=10 and d<1024+1. To run full experiment change lines 25+26 and 32+33. 
      Without these changes the standard deviations are a bit larger. You will also need to change xlim in plotting code 'plot.py'. 

    - The plot in the article uses sequential/parallel algorithm from https://github.com/zhangjiong724/
      Their code is not documented at all, it took us a week to get the code running on our machine. 
      To allow our code to be easily run, we chose not to use their code in the public version of our experiment. 
      Instead, we use a PyTorch implementation of the sequential algorithm and do not include the parallel algorithm.  
      This may cause some differences in the timing of the sequential algorithm relative to the article. 

"""

import numpy as np
import torch
import sys
import os 
import time 

from run_svd import run_svd, run_seq
from run_exp import run_exp, run_cay

import rotMatcuda as rotMatFn
import argparse

def getThetaCount(N, M):
    K = N-M
    return int(N*(N-1)/2) if K <= 1 else int(N*(N-1)/2) - int(K*(K-1)/2)

def experiment_rotMat(d, bs, G): 
    nThetas = getThetaCount(d,d)
    thetas  = torch.zeros(nThetas). normal_(0, np.pi)
    thetas.requires_grad_(True)

    X  = torch.zeros((d, bs)).normal_(0, 1)
    torch.cuda.synchronize()

    # Start timing of forward and backwards pass. 
    t0 = time.time()
    UX = rotMatFn.forward(X, thetas)
    rotMatFn.backward(thetas, UX, G)
    torch.cuda.synchronize()

    return time.time() - t0

def experiment_rotMatTeamRR(d, bs, G): 
    nThetas = getThetaCount(d,d)
    thetas  = torch.zeros(nThetas). normal_(0, np.pi)
    thetas.requires_grad_(True)

    X  = torch.zeros((d, bs)).normal_(0, 1)
    torch.cuda.synchronize()

    # Start timing of forward and backwards pass. 
    t0 = time.time()
    UX = rotMatFn.forwardTeamRR(X, thetas)
    rotMatFn.backwardTeamRR(thetas, UX, G)
    torch.cuda.synchronize()
    return time.time() - t0


def run_rotMat(d, bs, repeats, teamRR): 
    times = [] 
    G = torch.zeros((d, bs)).normal_(0, 1)  
    
    rotMatVersion = experiment_rotMatTeamRR if teamRR else experiment_rotMat
    for i in range(repeats + 1):
        t = rotMatVersion(d, bs, G)
        if i > 0: times.append(t)

    return np.array(times)


    
  
def main(bs, repeats, exclude_cayley, exclude_fasth):

    print("batch size:", bs, "repeats:", repeats)
    print("| %-10s | %-10s | %-10s | %-10s | %-10s |"%("dimension", "FastH", "Cayley", "RotMat", "RotMat TRR"))
 
    data = np.zeros((48, 5, repeats)) 
 
    for i, d in enumerate(range(64, 64*48+1, 64)): 
        svd = run_svd(d, bs, repeats)
        cay = run_cay(d, bs, repeats)
        rotMat = run_rotMat(d, bs, repeats, False)
        rotMatTeamRR = run_rotMat(d, bs, repeats, True)

        if not exclude_fasth: data[i, 0, :] = svd
        if not exclude_cayley: data[i, 1, :] = cay
        data[i, 2, :] = rotMat
        data[i, 3, :] = rotMatTeamRR

        print("| %-10i | %-10f | %-10f | %-10f | %-10f |"%(d, data[i, 0, :].mean(), data[i, 1, :].mean(), data[i, 2, :].mean(), data[i, 3, :].mean()))
        np.savez("rotMatComparison_" + str(bs), data)
 
        import plot
        plot.plot(exclude_cayley, exclude_fasth)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accept Batch Size and Trial Count Parametes and Benchmark RotMat")
    parser.add_argument("--bs", type=int, required=True, help="First argument Batch Size")
    parser.add_argument("--repeats", type=int, required=True, help="Second argument Trial Count")
    parser.add_argument("--exCayley", type=bool, required=False, help="Third optional argument to exclude Cayley Transform")
    parser.add_argument("--exFasth", type=bool, required=False, help="Third optional argument to exclude Fasth Algorithm")
    
    args = parser.parse_args()
    main(bs, args.repeats, args.exCayley, args.exFasth)

