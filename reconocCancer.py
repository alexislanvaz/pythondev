from matplotlib.cbook import flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


import cv2
import numpy as np
from torch import binary_cross_entropy_with_logits

modelo = Sequential()
modelo.add(Convolution2D(32,(3,3),input_shape=(224,224,3),activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))

modelo.add(Flatten())
modelo.add(Dense(128,activation='relu'))
modelo.add(Dense(50,activation='relu'))
modelo.add(Dense(1,activation='sigmoid'))

modelo.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurracy'])

# ------------------------------------------------------Fin primer video--------------------------------------

x_train =[]
y_train=[]
x_test=[]
y_test=[]
dataTr =[]

import glob
import os
for filename in glob.glob(os.path.join('/data/data/train/malignant','*.jpg')):
    dataTr.append([1,cv2.imread(filename)])

for filename in glob.glob(os.path.join('/data/data/train/benign','*.jpg')):
    dataTr.append([0,cv2.imread(filename)])

from random import shuffle
shuffle(dataTr)

for i,j in dataTr:
    y_train.append(i)
    x_train.append(j)

x_train = np.array(x_train)
y_train = np.array(y_train)

for filename in glob.glob(os.path.join('/data/data/test/malignant','*.jpg')):
    x_test.append(cv2.imread(filename))
    y_test.append(1) #Cancer
for filename in glob.glob(os.path.join('/data/data/test/benign','*.jpg')):
    x_test.append(cv2.imread(filename))
    y_test.append(0) #Maligno

x_test = np.array(x_test)
y_test = np.array(y_test)
modelo.fit(x_train,y_train,batch_size=32,epochs=4,validation_data=(x_test,y_test))

# -------------------------------------------------------------- SEGUNDO VIDEO---------------------------------------

ruta = 'data/data/test/malignant/276.jpg'

i = cv2.imread(ruta)

if round(modelo.predict(np.array([1]))[0][0]) == 1:
    print(f'La lesion es Cancer!!')
    cv2.imshow('Cancer',i)
else:
    print(f'La lesion es Benigna!!')
    cv2.imshow('Benigna',i)


# # Exportar el modelo a 'SavedModel'
# keras.experimental.export_saved_model(model, 'path_to_saved_model')