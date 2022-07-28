# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import rotMatcuda as rotMatFn
import torch.cuda.profiler as profiler



device = torch.device('cuda')
dtype = torch.float32

resultsPath = '/home/cangoksen14/funStuff/Givens-Orthogonal-Backprop/rotMat/'
resultsFname = 'fig3ComparisonIn512and256increments'

if not os.path.isdir ( resultsPath ):
    raise Exception("Non-existent results path!")

Nmin = 512
Nmax = 2048
Nstep = 256
Ns = range(Nmin,Nmax + 1,Nstep)
nNs = len(Ns)
bs = 32
nTrials = 100

forwardMilliseconds = torch.zeros(nNs)
backwardMilliseconds = torch.zeros(nNs)

profiler.start()
for i,N in enumerate(Ns):

    #print("On N={0:d}; largest is {1:d}".format(N,Nmax) )

    M = N
    K = N-M
    nThetas = int(N*(N-1)/2)

    # To drop angle params we need to have at least the last *pair* of dimensions left out
    if K > 1:
        nThetas -= int(K*(K-1)/2)
    #print("nThetas=" +str(nThetas) )

    #G = torch.randn(N,N,requires_grad=True).to(dtype).to(device)

    for t in range(nTrials+1):
        thetas = torch.randn(nThetas,requires_grad=True).to(dtype).to(device)
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

        startFwd = torch.cuda.Event(enable_timing=True)
        endFwd = torch.cuda.Event(enable_timing=True)
        
        startFwd.record()
        U = rotMatFn.forward(X,thetas)
        torch.cuda.synchronize()
        JVP = rotMatFn.backward(thetas,U,G)
        torch.cuda.synchronize()
        endFwd.record()

        forwardMilliseconds[i] += startFwd.elapsed_time(endFwd)
    
    torch.cuda.synchronize()
    forwardMilliseconds[i] /= nTrials
    print('On N={0:d}; With bs {1:d}  time in ms: {2:.10f} '.format(N, bs, startFwd.elapsed_time(endFwd)/1000) ) # milliseconds
    
profiler.stop()

#forwardMilliseconds /= nTrials
#backwardMilliseconds /= nTrials

cudaResults = {'Ns': list(Ns), 'fwd': forwardMilliseconds, 'bck': backwardMilliseconds}
torch.save( cudaResults, os.path.join(resultsPath,resultsFname) )
