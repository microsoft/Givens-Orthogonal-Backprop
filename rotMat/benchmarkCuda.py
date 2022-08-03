# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import rotMatcuda as rotMatFn
import torch.cuda.profiler as profiler

device = torch.device('cuda')
dtype = torch.float32

resultsPath = ''
resultsFname = ''

if not os.path.isdir ( resultsPath ):
    raise Exception("Non-existent results path!")

Nmin = 512
Nmax = 4096

Nstep = 256
Ns = range(Nmin,Nmax + 1,Nstep)
nNs = len(Ns)
bs = 32
nTrials = 100

def calculateThetaCount(N,M):
    K = N-M
    nThetas = int(N*(N-1)/2)
    # To drop angle params we need to have at least the last *pair* of dimensions left out
    if K > 1: nThetas -= int(K*(K-1)/2)
    return nThetas

forwardAndBackMilliSeconds = torch.zeros(nNs)

profiler.start()
for i,N in enumerate(Ns):
    #print("On N={0:d}; largest is {1:d}".format(N,Nmax) )
    nThetas = calculateThetaCount(N,N)
    #print("nThetas", nThetas)

    for t in range(nTrials+1):
        thetas = torch.randn(nThetas,requires_grad=True).to(dtype).to(device)
        X = torch.zeros((N, bs)).normal_(0, 1).to(dtype).to(device)
        G = torch.zeros((N, bs)).normal_(0, 1).to(dtype).to(device)
        torch.cuda.synchronize()

    for t in range(nTrials+1):
        thetas = torch.randn(get,requires_grad=True).to(dtype).to(device)
        X = torch.zeros((N, bs)).normal_(0, 1).to(dtype).to(device)
        G = torch.zeros((N, bs)).normal_(0, 1).to(dtype).to(device)
        torch.cuda.synchronize()
        # You *cannot* use time.time() to time cuda-enabled functions! The cpu
        # proceeds asynchronously, leading to ludicrous underestimates.
        #
        # The elapsed_time function returns time in *milliseconds*
        if t == 0:
            # warm up
            U = rotMatFn.forward(X,thetas)
            torch.cuda.synchronize()
            JVP = rotMatFn.backward(thetas,U,G)
            torch.cuda.synchronize()
            continue

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        U = rotMatFn.forward(X,thetas)
        torch.cuda.synchronize()
        JVP = rotMatFn.backward(thetas,U,G)
        torch.cuda.synchronize()
        end.record()

        torch.cuda.synchronize()
        forwardAndBackMilliSeconds[i] += start.elapsed_time(end)

    forwardAndBackMilliSeconds[i] /= nTrials
    print('On N={0:d}; With bs {1:d}  time in seconds: {2:.10f} '.format(N, bs, start.elapsed_time(end)/1000) ) # milliseconds
    
profiler.stop()
cudaResults = {'Ns': list(Ns), 'forwardAndBackMilliSeconds': forwardAndBackMilliSeconds}
torch.save( cudaResults, os.path.join(resultsPath,resultsFname) )
