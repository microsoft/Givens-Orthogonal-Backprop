# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import os
import torch
import rotMatcuda as rotMatFn
import time

device = torch.device('cuda')
dtype = torch.float32

N = 2000
M = N
K = N-M
nThetas = int(N*(N-1)/2)
batch_size =  32

# To drop angle params we need to have at least the last *pair* of dimensions left out
if K > 1:
    nThetas -= int(K*(K-1)/2)
print("N:",N, "| M:", M, "| #thetas:", nThetas, "| batch size:", batch_size,"\n")

X = torch.randn(N,batch_size).to(dtype).to(device)
G = torch.randn(X.size(0), X.size(1),requires_grad=True).to(dtype).to(device)
thetas = torch.randn(nThetas,requires_grad=True).to(dtype).to(device)

for teamRR in [True, False]:

    if teamRR: print("Using Team RR Scheduling:") 
    else: print("Using Circle Method RR Scheduling:")
    forward = rotMatFn.forwardTeamRR if teamRR else rotMatFn.forward
    backward = rotMatFn.backward
    # You *cannot* use time.time() to time cuda-enabled functions! The cpu
    # proceeds asynchronously, leading to ludicrous underestimates.

    # The elapsed_time function returns time in *milliseconds*

    startFwd = torch.cuda.Event(enable_timing=True)
    endFwd = torch.cuda.Event(enable_timing=True)
    startFwd.record()
    U = forward(X,thetas)
    endFwd.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    # Report in milliseconds
    forwardMilliseconds = startFwd.elapsed_time(endFwd)
    print('Forward time: {0:.5f}'.format(forwardMilliseconds))

    startBck = torch.cuda.Event(enable_timing=True)
    endBck = torch.cuda.Event(enable_timing=True)
    startBck.record()
    jvp = backward(thetas,U,G)
    endBck.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    # Report in milliseconds
    backwardMilliseconds = startBck.elapsed_time(endBck)
    print('Backward time: {0:.5f}'.format(backwardMilliseconds), "\n")
