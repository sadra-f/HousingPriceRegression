from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from random import shuffle
import multiprocessing
import time
import numpy as np

def plot_multi_column_on_process(X, Y, sub_w, sub_h, title=None):
    process = multiprocessing.Process(target=plot_multi_col, args=(X, Y, sub_w, sub_h, title))
    process.start()
    return process
def plot_on_process(file_path, title=None):
    process = multiprocessing.Process(target=plot_line, args=(file_path, title))
    process.start()
    return process

def plot_multi_col(X, Y, sub_w, sub_h, title=None):
    fig, axis = plt.subplots(sub_w, sub_h, sharey='all')
    fig.tight_layout()
    if title:
        fig.canvas.manager.set_window_title(title)
    _colors = plt.rcParams["axes.prop_cycle"]()
    colors = []
    for i in range(len(X[0])):
        c = next(_colors)["color"]
        colors.append(c)
    shuffle(colors)
    for i in range(len(X[0])):
        axis[i%sub_w, i%sub_h].scatter(X.transpose()[i], Y, color=colors[i])
        axis[i%sub_w, i%sub_h].set_title(i)
        axis[i%sub_w, i%sub_h].set_ylabel('Price')
    plt.show()

def _update(i):
    Y = np.loadtxt("tmp/loss_history.txt")[1:]
    X = [i*100 for i in range(len(Y))]
    plt.cla()
    plt.plot(X, Y)
    plt.title("Linear Regression Loss vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.gcf().canvas.manager.set_window_title('Test')
def plot_line(file_path , title=None):
    ani = FuncAnimation(plt.gcf(), _update, interval=1000)
    plt.tight_layout()
    plt.show()
    
