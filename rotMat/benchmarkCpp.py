# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import torch
import rotMatcpp as rotMatFn


device = torch.device('cpu')
dtype = torch.float32

resultsPath = ''
resultsFname = ''

if not os.path.isdir ( resultsPath ):
    raise Exception("Non-existent results path!")

Nmin = 20
Nmax = 920
Nstep = 220
Ns = range(Nmin,Nmax + 1,Nstep)
nNs = len(Ns)

nTrials = 20

forwardMilliseconds = torch.zeros(nNs)
backwardMilliseconds = torch.zeros(nNs)

for i,N in enumerate(Ns):

    print("On N={0:d}; largest is {1:d}".format(N,Nmax) )

    M = N
    K = N-M
    nThetas = int(N*(N-1)/2)
    
    # To drop angle params we need to have at least the last *pair* of dimensions left out
    if K > 1:
        nThetas -= int(K*(K-1)/2)
    print("nThetas=" +str(nThetas) )

    G = torch.randn(N,N,requires_grad=True).to(dtype).to(device)
    thetas = torch.randn(nThetas,requires_grad=True).to(dtype).to(device)

    for t in range(nTrials):

        start = time.time()
        U = rotMatFn.forward(thetas,N)
        elapsedFwd = time.time()-start
        start = time.time()
        JVP = rotMatFn.backward(thetas,U,G)
        elapsedBck = time.time()-start

        # Milliseconds
        forwardMilliseconds[i] += elapsedFwd*1000.
        backwardMilliseconds[i] += elapsedBck*1000.

        print('Forward time: {0:.3f} | Backward time: {1:.3f}'.format(elapsedFwd, elapsedBck)) # milliseconds

forwardMilliseconds /= nTrials
backwardMilliseconds /= nTrials

cppResults = {'Ns': list(Ns), 'fwd': forwardMilliseconds, 'bck': backwardMilliseconds}
torch.save( cppResults, os.path.join(resultsPath,resultsFname) )
