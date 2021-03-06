import numpy as np
from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self, layer_number):
        self.layer_number = layer_number
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model three layer neuron, with 3 input connections and 3 output connections for the first two layers, and with 3 input connections and 1 output connection for the last layer.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = []
        for i in range(self.layer_number - 1):
            self.synaptic_weights.append(2 * random.random((3, 3)) - 1)

        self.synaptic_weights.append(2 * random.random((3, 1)) - 1)


    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    # Coded according to https://en.wikipedia.org/wiki/Backpropagation
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        learning_rate = 0.1
        for iteration in range(number_of_training_iterations):

            outputs = self.think(training_set_inputs)
            
            error_last = training_set_outputs - outputs[2].T
            delta_last = error_last * self.__sigmoid_derivative(outputs[2].T)
            adjust_last = learning_rate * dot(outputs[1], delta_last)

            error_second_last = dot(delta_last, self.synaptic_weights[2].T)
            delta_second_last = error_second_last * self.__sigmoid_derivative(outputs[1].T)
            adjust_second_last = learning_rate * dot(outputs[0], delta_second_last)


            error_third_last = dot(delta_second_last, self.synaptic_weights[1].T)
            delta_third_last = error_third_last * self.__sigmoid_derivative(outputs[0].T)
            adjust_third_last = learning_rate * dot(training_set_inputs.T, delta_third_last)

            self.synaptic_weights[0] += adjust_third_last
            self.synaptic_weights[1] += adjust_second_last
            self.synaptic_weights[2] += adjust_last



    # The neural network thinks.
    def think(self, inputs):
        all_outputs = []
        hidden_inputs = inputs
        for i in range(self.layer_number):
            # Pass inputs through our neural network.
            hidden_outputs = self.__sigmoid(dot(hidden_inputs, self.synaptic_weights[i]))
            all_outputs.append(hidden_outputs.T)
            hidden_inputs = hidden_outputs
        
        return all_outputs


if __name__ == "__main__":

    #Intialise a 3 layer neuron neural network.
    layer_number = 3
    neural_network = NeuralNetwork(layer_number)

    print("Random starting weights: ")
    print(neural_network.synaptic_weights)
    #print(neural_network.last_weights)
    
    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New weights after training: ")
    print(neural_network.synaptic_weights)
    #print(neural_network.last_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0]))[2])
    
