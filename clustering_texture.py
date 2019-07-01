import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate

n_dimensions = 40
CLASSES = 30
lle_neighbours = 60
KNNeis = 5

X = np.load("./" + str(lle_neighbours) +"_neisTexture_c"+ str(CLASSES) +"_d"+str(n_dimensions) + ".npy") 
Y = np.load("./tex"+str(CLASSES) + "Imglabels.npy")

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.3)

model = KMeans(n_clusters=30)
# kFold = model_selection.KFold(n_splits=10)
# cv_results = model_selection.cross_val_score(model,X_train,Y_train, cv= kFold, scoring = 'accuracy')

# cv_results = np.around(cv_results,decimals = 4)
# print("cv_results : " ,cv_results)

model = model.fit(X_train) # NO Y_tran?
labels = model.predict(X_train)

act_labels = []
print(len(X_train)- len(Y_train))
for i in range(len(X_train)):
    act_labels.append(np.argmax(Y_train[i]))
df = pd.DataFrame({'labels': labels,'Data' : act_labels})
df_new = df.sort_values('labels')
np.savetxt(r'results_KMeans.txt', df_new.values, fmt='%d')
Pred = model.predict(X_test)
print(df_new.head(100))
print(model.inertia_)
# acc = accuracy_score(Y_test,Pred)
# acc = round(acc,4)
# print(acc)
# #print(classification_report(Y_test,Pred))

file1 = open("res_inertia_KMeans.txt","a") 
# # table = [[CLASSES,n_dimensions,lle_neighbours,KNNeis,acc]]
file1.write("\n")
file1.write(str(CLASSES) + "        "+ str(n_dimensions) +"        "+ str(lle_neighbours)+"        "+ str(model_inertia) + "      ")
# file1.write(str(df.sort_values("labels")))
file1.close()