# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from numpy import ndarray as nd
from pacman import Directions
import math
from featureExtractors import FEATURE_NAMES

PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood', 
            'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            "foodCount",
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',  
            'furthestFood', 
            'numberAvailableActions',
            "ratioCapsuleDistance",
            "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance"
            ]
        
        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))

        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        hidden_sizes = []
        input_size = len(feature_names_to_use)
        output_size = 1

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        if hidden_sizes:
            self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
            self.biases.append(np.zeros((1, hidden_sizes[0])))

            # Hidden layers
            for i in range(1, len(hidden_sizes)):
                self.weights.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * 0.01)
                self.biases.append(np.zeros((1, hidden_sizes[i])))

            # Last hidden layer to output layer
            self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
            self.biases.append(np.zeros((1, output_size)))
        else:
            self.weights.append(np.random.randn(input_size) * 0.01)
            self.biases.append(np.zeros((1, 1)))

    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        np.maximum(0,x)

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activation = [X]
    
        for i in range(len(self.weights)):
            # linear transformation 
            z = np.dot(self.activation[-1], self.weights[i]) + self.biases[i]

            # apply activation function depending on different layer
            if i < len(self.weights) -1:
                a = self.activationHidden(z)
            else:
                a = self.activationOutput(z)
            
            self.activation.append(a[0])

        return self.activation[-1]

    def backward(self, X:nd, y:nd):

        prediction = self.activation[-1]
        m = y.shape[0]

        # compute the output error
        output_error = prediction - y

        # compute gradients for the output layer
        sigmoid_derivative = prediction * (1 - prediction)
        output_d = output_error * sigmoid_derivative

        dw = [np.dot(self.activation[-2].T, output_d) / m]  # Weights for output layer
        db = [np.sum(output_d) / m] 

        # backword pass for hidden layer (if any)
        for i in range(len(self.weights)-1, 0, -1):
            prediction = self.activation[i]

            error = np.dot(output_d, self.weights[i].T)
            hidden_d = error * np.where(prediction > 0, 1, 0)

            dw.append( np.dot(self.a[i-1].T, hidden_d) / m )
            db.append(np.sum(hidden_d, axis=0, keepdims=True) / m )

            output_d = hidden_d
        
        dw.reverse()
        db.reverse()

        return dw, db

    def update_weights(self, dw, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train: np.ndarray = trainingData[:, self.features_to_use]
        X_validate = validationData[:, self.features_to_use]

        for epoch in range(self.max_iterations):
            prediction = self.forward(X_train[:, 1:])
            dw, db = self.backward(X_train[:, 1:], trainingLabels)
            self.update_weights(dw,db)       

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        predictions = self.forward(vector_to_classify[1:])
        # return [1 if predictions >= 0.5 else 0 ]
        return predictions

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here. 
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        predictions = self.forward(X_eval[:, 1:])
        loss = np.mean((predictions - labels) ** 2) # mean square error
        print(f"Loss: {loss: .4f}")

        result = [1 if p >= 0.5 else 0 for p in predictions]

        return f"Expectation: {labels} \nResult: {result}"

    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        with open(weights_path, 'w') as f:
            for layer_idx in range(len(self.weights)):
                # Save weights
                np.savetxt(f, self.weights[layer_idx], header=f'Weights Layer {layer_idx}', comments='')
                # Save biases
                np.savetxt(f, self.biases[layer_idx], header=f'Biases Layer {layer_idx}', comments='')

    def load_weights(self, weights_path):
        """
        Loads your weights from a .model file. 
        Whatever you do here should work with the formatting of your save_weights function.
        """
        with open(weights_path, 'r') as f:
            lines = f.readlines()
            i = -1
            result = [[],[]]
            current_list = []  # Keeps track of whether we're appending to weights or biases

            for line in lines:
                line = line.strip()

                try:
                    value = float(line)
                    current_list.append(value) 
                except:
                    if i >= 0:
                        result[i%2].append(current_list)
                        current_list = []

                    if 'Weights Layer' in line:
                        i += 1
                        continue
                    elif 'Biases Layer' in line:
                        i += 1
                        continue
            
            result[i%2].append(current_list)

        self.weights, self.biases =  result[0], result[1]   
