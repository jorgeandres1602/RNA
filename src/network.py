'''Red neuronal con keras'''

'''Importar librerias'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
import numpy as np

'''Definir parámetros'''
learning_rate = 3
epochs = 30
batch_size = 10

''''Cargar los datos'''
dataset=mnist.load_data()

'''Colocar los datos en la forma adecuada para leerlos'''
dat=np.array(dataset)
print(dat[1,1].shape)
(x_train, y_train), (x_test, y_test) = dataset

'''Normalizar las imagenes'''
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255

'''Definir el tamaño de la salida de nuestra red'''
num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

'''Añadir las capas de la red'''
model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(Dense(num_classes, activation='sigmoid'))

'''Cargar la red'''
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])

'''Iniciar el entrenamiento'''
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )

model.summary()
