import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate

n_dimensions = 60
CLASSES = 45
lle_neighbours = 60
KNNeis = 5

X = np.load("./" + str(lle_neighbours) +"_neisTexture_c"+ str(CLASSES) +"_d"+str(n_dimensions) + ".npy") 
Y = np.load("./tex"+str(CLASSES) + "Imglabels.npy")

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.3)

model = KNeighborsClassifier(n_neighbors= KNNeis)
kFold = model_selection.KFold(n_splits=10)
cv_results = model_selection.cross_val_score(model,X_train,Y_train, cv= kFold, scoring = 'accuracy')

cv_results = np.around(cv_results,decimals = 4)
print("cv_results : " ,cv_results)

model = model.fit(X_train,Y_train)
Pred = model.predict(X_test)
acc = accuracy_score(Y_test,Pred)
acc = round(acc,4)
print(acc)
#print(classification_report(Y_test,Pred))

file1 = open("results45.txt","a") 
table = [[CLASSES,n_dimensions,lle_neighbours,KNNeis,acc]]
file1.write("\n")
file1.write(str(CLASSES) + "        "+ str(n_dimensions) +"        "+ str(lle_neighbours)+"        "+str(KNNeis)+"        " + str(acc) + "      "+ str(cv_results))
file1.close()