import math
import os
import torch
import rotMatcuda as rotMatFn
import torch.cuda.profiler as profiler


threadsPerBlock = 256;

device = torch.device('cuda')
dtype = torch.float32

resultsPath = ''
resultsFname = ''

if not os.path.isdir ( resultsPath ):
    raise Exception("Non-existent results path!")

Nmin = 20
Nmax = 2000
Nstep = 220
Ns = range(Nmin,Nmax + 1,Nstep)
nNs = len(Ns)

nTrials = 50

forwardMilliseconds = torch.zeros(nNs)
backwardMilliseconds = torch.zeros(nNs)

profiler.start()
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

        # You *cannot* use time.time() to time cuda-enabled functions! The cpu
        # proceeds asynchronously, leading to ludicrous underestimates.
        #
        # The elapsed_time function returns time in *milliseconds*

        startFwd = torch.cuda.Event(enable_timing=True)
        endFwd = torch.cuda.Event(enable_timing=True)
        startFwd.record()
        U = rotMatFn.forward(thetas,N,threadsPerBlock)
        endFwd.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        startBck = torch.cuda.Event(enable_timing=True)
        endBck = torch.cuda.Event(enable_timing=True)
        startBck.record()
        JVP = rotMatFn.backward(thetas,U,G,threadsPerBlock)
        endBck.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        # Milliseconds
        forwardMilliseconds[i] += startFwd.elapsed_time(endFwd)
        backwardMilliseconds[i] += startBck.elapsed_time(endBck)
        print('Forward time: {0:.3f} | Backward time: {1:.3f}'.format(startFwd.elapsed_time(endFwd), startBck.elapsed_time(endBck)) ) # milliseconds

profiler.stop()

forwardMilliseconds /= nTrials
backwardMilliseconds /= nTrials

cudaResults = {'Ns': list(Ns), 'fwd': forwardMilliseconds, 'bck': backwardMilliseconds}
torch.save( cudaResults, os.path.join(resultsPath,resultsFname) )
