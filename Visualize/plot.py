from matplotlib import pyplot as plt
from random import shuffle
import multiprocessing

def plot_on_thread(X, Y, sub_w, sub_h):
    process = multiprocessing.Process(target=plot_multi_col, args=(X, Y, sub_w, sub_h))
    process.start()
    return process

def plot_multi_col(X, Y, sub_w, sub_h):
    fig, axis = plt.subplots(sub_w, sub_h, sharey='all')
    fig.tight_layout()

    _colors = plt.rcParams["axes.prop_cycle"]()
    colors = []
    for i in range(len(X.columns)):
        c = next(_colors)["color"]
        colors.append(c)
    shuffle(colors)
    for i, col in enumerate(X.columns):
        axis[i%sub_w, i%sub_h].scatter(X[col], Y, color=colors[i])
        axis[i%sub_w, i%sub_h].set_title(col)
        axis[i%sub_w, i%sub_h].set_ylabel('Price')
    plt.show()