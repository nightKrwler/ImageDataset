import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import locally_linear_embedding

X = np.load("./train.npy") 
Y = np.load("./train_labels.npy")
colors = ['k', 'c', 'r', 'g', 'b', 'y', 'm', 'darkgreen', 'indigo', 'lightpink']

for neis in [25]:
    (data_lle, _) = locally_linear_embedding(X, n_neighbors = neis, n_components = 2)
    np.save(str(neis)+'-neis1',data_lle)
    plt.figure()
    
    X = data_lle
    for i in range(len(X)):
        plt.scatter(X[i,0], X[i,1], s = 1, c = colors[np.argmax(Y[i])], marker = "o")
    plt.show()
