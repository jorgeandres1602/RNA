# -*- coding: utf-8 -*-
"""
network.py
~~~~~~~~~~

Este código se trata de una red neuronal que usa el algoritmo Stochastic Gradient Descent para el 
reconocimiento de dígitos a través de imagenes vectorizadas en la escala de grises..
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """En ésta funcion se crea el vector de los bias junto con la matriz de los pesos con números random."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.parameters = {'biases': self.biases, 'weights': self.weights}

    def feedforward(self, a):
        """En esta función se evalúa 'a' en la funcion de activación."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

     def RMSprop(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Optimizador alternativo RMSprop"""
        
        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)
        training_data=list(training_data)
        n = len(training_data)
        epsilon = 1e-8  #Parámetros
        beta = 0.9
        squared_gradients = {param_name: np.zeros_like(param_name) for param_name in self.parameters}      
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                for param_name in self.parameters.keys():
                    gradient_sum = np.zeros_like(self.parameters[param_name])
                    for x, y in mini_batch:
                        output = self.feedforward(x)
                        output_softmax = softmax(output)#Capa softmax
                        x_grad = self.backprop(x, y)
                        gradient_sum += x_grad['nabla_b'][param_name]
                    gradient = gradient_sum / len(mini_batch)
                    squared_gradients[param_name] = beta * squared_gradients[param_name] + (1 - beta) * (gradient ** 2)
                    self.parameters[param_name] -= (eta / (np.sqrt(squared_gradients[param_name]) + epsilon)) * gradient
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

   def update_mini_batch(self, mini_batch, eta):
        """En ésta funcion se actualizan los valores de los bias y los pesos con el algoritmo de Stochastic Gradient Descent."""
        nabla_b = {param_name: np.zeros(b.shape, dtype=float) for param_name, b in enumerate(self.biases)}#Cambio de formato a diccionarios en lugar de listas
        nabla_w = {param_name: np.zeros(w.shape, dtype=float) for param_name, w in enumerate(self.weights)}#Se intentó colocar dtype=float pero no solucionó el problema
        for x, y in mini_batch:
            delta_nabla = self.backprop(x, y)
            for param_name in self.parameters.keys():
                nabla_b[param_name].append(delta_nabla['nabla_b'][param_name])
                nabla_w[param_name].append(delta_nabla['nabla_w'][param_name])

    
        for param_name in self.parameters.keys():
            self.weights[param_name] -= (eta / len(mini_batch)) * nabla_w[param_name]
            self.biases[param_name] -= (eta / len(mini_batch)) * nabla_b[param_name]

    def backprop(self, x, y):#función modificada para implementar elc ross entropy
        """Regresa un vector de dos dimensiones que contiene a nabla b y nabla w que representan el gradiente de la función de costo."""
        nabla_b = {param_name: np.zeros(b.shape) for param_name, b in enumerate(self.biases)} #Diccionarios de nabla para operarlos en el mini batch
        nabla_w = {param_name: np.zeros(w.shape) for param_name, w in enumerate(self.weights)}
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return {'nabla_b': nabla_b, 'nabla_w': nabla_w}

    def evaluate(self, test_data):
        """Regresa el número de imágenes que adivinó correctamente."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Devuelve el vector de las derivadas parciales."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """Representa la función de activación (función sigmoide)."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """La derivada de la función de activación."""
    return sigmoid(z)*(1-sigmoid(z))
#Función para la capa softmax
def softmax(x):
    exp_x = np.exp(x-np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)
