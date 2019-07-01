import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import locally_linear_embedding

files = ["./10-neis.npy"]
Y = np.load("./train_labels.npy")
colors = ['k', 'c', 'r', 'g', 'b', 'y', 'm', 'darkgreen', 'indigo', 'lightpink']

for file in files:
    plt.figure()
    X = np.load(file) 
    for i in range(len(X)):
        plt.scatter(X[i,0], X[i,1], s = 1, c = colors[np.argmax(Y[i])], marker = "o")
    plt.show()