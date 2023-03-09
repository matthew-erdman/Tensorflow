import numpy as np
import matplotlib.pyplot as plt

def graph():
    tr_ls = []
    tr_ac = []
    vl_ls = []
    vl_ac = []
    with open('logs/log', 'r') as input_file:  # logged via Pycharm
        for line in input_file.readlines():
            if 'loss:' in line:
                parts = line.split()
                if float(parts[5]) < 2:
                    tr_ls.append(float(parts[5])/2)
                else:
                    tr_ls.append(1.)
                tr_ac.append(float(parts[8]))
                if float(parts[11]) < 2:
                    vl_ls.append(float(parts[11])/2)
                else:
                    vl_ls.append(1.)
                vl_ac.append(float(parts[14]))

    epochs = len(tr_ls)
    x = np.arange(epochs)

    fig, ax = plt.subplots()
    plt.plot(x, tr_ls[:epochs], color='red', label='Training Loss')
    plt.plot(x, tr_ac[:epochs], color='orange', label='Training Accuracy')
    plt.plot(x, vl_ls[:epochs], color='blue', label='Validation Loss')
    plt.plot(x, vl_ac[:epochs], color='green', label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.show()
