import matplotlib.pyplot as plt
import sys
import numpy as np

batch_size = 32
def plot(exclude_cayley=False, exclude_fasth=False): 
    blue    = "C0"
    orange  = "C1"
    green   = "C2"
    red     = "C3"
    purple  = "C4"

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.8))
    xs      = np.array(range(64, 64*48+1, 64))
    data    = np.load("rotMatComparison_" + str(batch_size) +".npz")['arr_0']
    
    def plot(data, name, limit=2048, color=None):
        if data.shape[0] < limit: limit = data.shape[0]
        mean = np.mean(data[:limit], 1) * 1000
        plt.plot(xs, mean, '-', label=name, color=color)
        plt.fill_between(xs[:limit], mean - np.std(data[:limit], 1), mean + np.std(data[:limit], 1), alpha=0.3, linewidth=0, color=color)
    
    if not exclude_fasth: plot(data[:, 0], "FastH",            color=blue)
    plot(data[:, 3], "RotMat Team RR",           color=green)
    plot(data[:, 2], "RotMat",           color=red)
    if not exclude_cayley: plot(data[:, 1], "Cayley",      color=purple)

    plt.legend()

    plt.xlabel("Size of matrix $d with batch size " + str(batch_size))
    plt.ylabel("Time in milliseconds ")
    
    if exclude_cayley and exclude_fasth:
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        plt.yticks(np.arange(0, ymax, 1))
    
    plt.tight_layout()
    plt.savefig("running_time_" + str(batch_size) + ".png")
    plt.show()

plot()
