import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import locally_linear_embedding


n_dimensions = 60
CLASSES = 45
lle_neis = 10
X = np.load("./tex"+ str(CLASSES) + "Img.npy") 


for neis in [lle_neis]:
    print("hello")
    (data_lle, _) = locally_linear_embedding(X, n_neighbors = neis, n_components = n_dimensions)
    np.save(str(neis)+'_neisTexture_c'+str(CLASSES) +'_d' + str(n_dimensions) ,data_lle)
    
