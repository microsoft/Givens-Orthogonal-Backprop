# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import os
import torch
import rotMatcuda as rotMatFn
import time

device = torch.device('cuda')
dtype = torch.float32

N = 2048
M = N
K = N-M
nThetas = int(N*(N-1)/2)

# To drop angle params we need to have at least the last *pair* of dimensions left out
if K > 1:
    nThetas -= int(K*(K-1)/2)
print("nThetas=" +str(nThetas) )

G = torch.randn(N,N,requires_grad=True).to(dtype).to(device)
thetas = torch.randn(nThetas,requires_grad=True).to(dtype).to(device)

# You *cannot* use time.time() to time cuda-enabled functions! The cpu
# proceeds asynchronously, leading to ludicrous underestimates.

# The elapsed_time function returns time in *milliseconds*

startFwd = torch.cuda.Event(enable_timing=True)
endFwd = torch.cuda.Event(enable_timing=True)
startFwd.record()
U = rotMatFn.forward(thetas,N)
endFwd.record()
torch.cuda.synchronize()
# Waits for everything to finish running
torch.cuda.synchronize()

# Report in milliseconds
forwardMilliseconds = startFwd.elapsed_time(endFwd)
print('Forward time: {0:.5f}'.format(forwardMilliseconds))

startBck = torch.cuda.Event(enable_timing=True)
endBck = torch.cuda.Event(enable_timing=True)
startBck.record()
JVP = rotMatFn.backward(thetas,U,G)
endBck.record()
# Waits for everything to finish running
torch.cuda.synchronize()

# Report in milliseconds
backwardMilliseconds = startBck.elapsed_time(endBck)
print('Backward time: {0:.5f}'.format(backwardMilliseconds))
