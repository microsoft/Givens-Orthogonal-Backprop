import matplotlib.pyplot as plt
import sys
import numpy as np

def plot(): 
    blue    = "C0"
    orange  = "C1"
    green   = "C2"
    red     = "C3"
    purple  = "C4"

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.8))
    xs      = np.array(range(64, 64*48+1, 64))#np.array(range(64, 2048 + 1, 64))
    data    = np.load("rotMatComparison.npz")['arr_0']
    
    def plot(data, name, limit=2048, color=None):
        if data.shape[0] < limit: limit = data.shape[0]
        mean = np.mean(data[:limit], 1)
        plt.plot(xs, mean, '-', label=name, color=color)
        plt.fill_between(xs[:limit], mean - np.std(data[:limit], 1), mean + np.std(data[:limit], 1), alpha=0.3, linewidth=0, color=color)
    
    #plot(data[:, 0], "FastH",            color=blue)
    plot(data[:, 3], "RotMat Team RR",           color=green)
    plot(data[:, 2], "RotMat",           color=red)
    #plot(data[:, 1], "Cayley",      color=purple)

    plt.legend()

    plt.xlabel("Size of matrix $d$")
    plt.ylabel("Time in seconds ")
    
    ymin, ymax = plt.ylim()
    print(ymin, ymax)
    plt.yticks(np.arange(0, ymax, 0.001))
    #plt.tight_layout()
    plt.savefig("running_time.png")
    plt.show()

plot()
