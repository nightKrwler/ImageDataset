import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

n_dimensions = 479
CLASSES = 30
lle_neighbours = 60


X = np.load("./imageArrays/" + str(lle_neighbours) +"_neisTexture_c"+ str(CLASSES) +"_d"+str(n_dimensions) + ".npy") 
Y = np.load("./tex"+str(CLASSES) + "Imglabels.npy")
label = []
for i in Y :
    label.append(np.argmax(i))

print(np.shape(label))
np.reshape(label,(-1,1))

print(np.shape(label))


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,label,test_size = 0.1)


print(label)


print( np.shape(Y_train))

model = Sequential([
    Dense(400, input_shape = (n_dimensions,), activation='relu'),
    Dense(300, activation='relu'),
    Dense(200, activation='relu'),
    Dense(100, activation='relu'),
    Dense(CLASSES, activation='softmax')
])

model.summary()

model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #decreasing learning rate is getting accuracy lower

scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform(X_train)
print(np.shape(scaled_train_samples))

model.fit(scaled_train_samples, np.array(Y_train).T, validation_split=0.1, batch_size=10, epochs=50, verbose=2)