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
        squared_gradients = {parameter: np.zeros_like(parameter) for parameter in self.parameters}      
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                for i, param_name in enumerate(self.parameters.keys()):
                    #print(f"beta: {beta}, type(beta): {type(beta)}")
                    #print(f"squared_gradients[param_name]: {squared_gradients[param_name]}, type(squared_gradients[param_name]): {type(squared_gradients[param_name])}")
                    gradient = (1 / len(mini_batch)) * np.sum([self.backprop(x, y)[0][i] for x, y in mini_batch])
                    #print(f"gradient: {gradient}, type(gradient): {type(gradient)}")
                    squared_gradients[param_name] = np.zeros_like(gradient, dtype=np.float64)
                    squared_gradients[param_name] = beta * squared_gradients[param_name] + (1 - beta) * (gradient ** 2)
                    self.parameters[param_name] -= (eta / (np.sqrt(squared_gradients[param_name]) + epsilon)) * gradient
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """En ésta funcion se actualizan los valores de los bias y los pesos con el algoritmo de Stochastic Gradient Descent."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Regresa un vector de dos dimensiones que contiene a nabla b y nabla w que representan el gradiente de la función de costo."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
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
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

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
