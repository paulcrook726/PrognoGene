import numpy as np
from Bio import SeqIO


def funcheck(item):
    if item is "A" or "G" or "T" or "C":
        return True
    else:
        return False
def activation(input_signal):
    """
    This function acts as the cellular activation function.  The sigmoid of input_signal is returned, although
    other types of functions exist which may be used instead.

    :param input_signal:

    :type input_signal:

    :return:

    :rtype:

    """
    return 1 / (1 + np.exp(-input_signal))  # sigmoid activation function


def activation_d(x):
    """
    This is the derivative of the sigmoid function.

    :param output:

    :type output:

    :return:

    :rtype:

    """

    return activation(x) * (1.0 - activation(x))  # getting input from output


class NeuralNet(object):
    """
    Blueprints for a neural network object.
    """

    def __init__(self, num_inputs, num_outputs, num_hidden_nodes):
        self.num_inputs = num_inputs + 1  # Bias node
        self.num_outputs = num_outputs
        self.hidden_size = num_hidden_nodes

        self.input_weight_matrix = np.random.rand(self.num_inputs, self.hidden_size)
        self.output_weight_matrix = np.random.rand(self.hidden_size, self.num_outputs)

        self.input_values = [1.0] * self.num_inputs
        self.output_values = [1.0] * self.num_outputs
        self.hidden_values = [1.0] * self.hidden_size

        self.input_change = np.zeros((self.num_inputs, self.hidden_size))
        self.output_change = np.zeros((self.hidden_size, self.num_outputs))

    def feed_forward(self, input_data):
        """

        :param input_data:
        :type input_data:
        :return:
        :rtype:
        """

        if len(input_data) != self.num_inputs - 1:
            raise Exception("Mismatching input parameters.  Check instance init data and method call data!")

        #  inputting data into neural net input matrix

        for input_node in range(self.num_inputs - 1):  # Bias node is ignored
            self.input_values[input_node] = input_data[input_node]

        #  calculating hidden matrix data via activation function

        for hidden_node in range(self.hidden_size):
            summation = 0.0  # value containing all the signals from every connected node
            #  ( nodes from the previous layer)
            for input_node in range(self.num_inputs):
                summation += self.input_values[input_node] * self.input_weight_matrix[input_node][hidden_node]
            self.hidden_values[hidden_node] = activation(summation)

        #  calculating output layer data via activation function (same as above, except operating on output layer)

        for output_node in range(self.num_outputs):
            summation = 0.0
            for hidden_node in range(self.hidden_size):
                summation += self.hidden_values[hidden_node] * self.output_weight_matrix[hidden_node][output_node]
            self.output_values[output_node] = activation(summation)

        return self.output_values[:]  # returns the calculation of the neural net's effects on the data set

    def back_propogation(self, true_outcomes, learn_rate):
        if len(true_outcomes) != self.num_outputs:
            raise Exception("Mismatching input parameters.  Check instance init data and method call data!")

        #  calculating error and direction of error between true outcome and predicted outcome layer values

        error_slope_out = [0.0] * self.num_outputs
        for output_node in range(self.num_outputs):
            error = -(true_outcomes[output_node] - self.output_values[output_node])
            error_slope_out[output_node] = activation_d(self.output_values[output_node]) * error

        #  calculating error and direction of error between output layer and hidden layer

        error_slope_hid = [0.0] * self.hidden_size
        for hidden_node in range(self.hidden_size):
            error = 0.0
            for output_node in range(self.num_outputs):
                error += error_slope_out[output_node] * self.output_weight_matrix[hidden_node][output_node]
            error_slope_hid[hidden_node] = activation_d(self.hidden_values[hidden_node] * error)

        #  input new weights based on error and slope: output values --> hidden values

        for hidden_node in range(self.hidden_size):
            for output_node in range(self.num_outputs):
                change = error_slope_out[output_node] * self.hidden_values[hidden_node]
                self.output_weight_matrix[hidden_node][output_node] -= learn_rate * \
                                                                       change + \
                                                                       self.output_change[hidden_node][output_node]
                self.output_change[hidden_node][output_node] = change

        #  new weights: hidden values --> input values

        for input_node in range(self.num_inputs):
            for hidden_node in range(self.hidden_size):
                change = error_slope_hid[hidden_node] * self.input_values[input_node]
                self.input_weight_matrix[input_node][hidden_node] -= learn_rate * \
                                                                     change + \
                                                                     self.input_change[input_node][hidden_node]
                self.input_change[input_node][hidden_node] = change\

        #  calculate new error

        error = 0.0
        for i in range(len(true_outcomes)):
            error += 0.5 * (true_outcomes[i] - self.output_values[i]) ** 2

        return error

    def train(self, training_data, iterations=3000, learn_rate=0.002):
        for i in range(iterations):
            error = 0.0
            for data_point in training_data:
                inputs = data_point[0]
                desired = data_point[1]
                self.feed_forward(inputs)
                error = self.back_propogation(desired, learn_rate)
            if i % 500 == 0:
                print("Error: " + str(error) + "\n")

with open("coding_sequence_data\chr_1_cs.txt") as f:
    lines = [line.strip('\n') for line in f]
str_lines = ''.join(lines)
coding_genes = str_lines.split(">")
coding_genes = [item for item in coding_genes if funcheck(item)]
genes = []
for gene in coding_genes:
    gene_data = []
    for letter in gene:
        if letter == "A":
            gene_data.append(0.1260)
        elif letter == "C":
            gene_data.append(0.1340)
        elif letter == "G":
            gene_data.append(0.0806)
        elif letter == "T":
            gene_data.append(0.1335)
        else:
            pass
    genes.append([gene_data])
    for gene in genes:
        print(len(gene[0]))








