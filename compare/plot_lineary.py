import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(bs, exclude_cayley=False, exclude_fasth=False, compareLinear=False):
    blue    = "C0"
    orange  = "C1"
    green   = "C2"
    red     = "C3"
    purple  = "C4"

    # Data preparation
    batch_size = int(bs)
    xs      = np.array(range(64, 64*48+1, 64))
    data    = np.load("rotMatComparison_" + str(batch_size) +".npz")['arr_0']

    # Line plot
    fig, ax = plt.subplots(figsize=(8, 6))

    def plot(data, name, limit=2048, color=None):
        if data.shape[0] < limit: limit = data.shape[0]
        mean = np.mean(data[:limit], 1) * 1000
        plt.plot(xs, mean, '-', label=name, color=color)
        plt.fill_between(xs[:limit], mean - np.std(data[:limit], 1), mean + np.std(data[:limit], 1), alpha=0.3, linewidth=0, color=color)

    if not exclude_fasth:
        plot(data[:, 0], "FastH", color=blue)

    plot(data[:, 3], "RotMat Team RR", color=green)
    plot(data[:, 2], "RotMat", color=red)

    if not exclude_cayley:
        plot(data[:, 1], "Cayley", color=purple)

    plt.xlabel("Size of matrix $d$ with batch size " + str(batch_size), fontsize=14)
    plt.ylabel("Time in milliseconds", fontsize=14)
    plt.xlim(left=xs[0], right=xs[-1])

    if exclude_cayley and exclude_fasth:
        plt.ylim(0, plt.ylim()[1])
        plt.yticks(np.arange(0, plt.ylim()[1], 1))
    else:
        plt.ylim(0, None)

    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig("running_time_line_linearscale_" + str(batch_size) + ".png", dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        bs = sys.argv[1]
        plot(bs)
    else:
        print("Please provide a batch size as a command line argument.")
