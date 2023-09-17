'''Red neuronal con keras'''

'''Experimento 2 con regularización L1'''

'''Instalar comet e importarlo'''
%pip install comet_ml

import comet_ml
comet_ml.init(project_name="Experimentos tarea 3")

'''Importar librerias'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
import numpy as np

'''Parámetros comet'''
experiment = comet_ml.Experiment(
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    log_code=True,
)
'''Definir parámetros'''
import numpy as np
learning_rate=2

parameters = {
    "batch_size": 100,
    "epochs": 30,
    "optimizer": "rmsprop",
    "loss": "categorical_crossentropy",
}

experiment.log_parameters(parameters)

''''Cargar los datos'''
dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # x_trainv = x_trainv/255
x_test /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

'''Añadir las capas de la red'''
model = Sequential()
model.add(Input(shape=(28,28))) 
model.add(Flatten()) 
model.add(Dense(150, activation='relu')) 
model.add(Dense(200, activation='selu', kernel_regularizer=regularizers.L1(0.01)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

model.summary()

'''Seleccionar ubicación para guardar el modelo'''
filepath = "best_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
'''Cargar la red'''
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])

'''Iniciar el entrenamiento'''
model.fit(x_train, y_trainc,
                    batch_size=parameters['batch_size'],
                    epochs=parameters["epochs"],
                    verbose=1,
                    validation_data=(x_test, y_testc),
                    callbacks=[checkpoint])
'''Evaluar el modelo'''
score = model.evaluate(x_test, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)

'''Guardar el modelo'''
model.save("97Ef.h5")
