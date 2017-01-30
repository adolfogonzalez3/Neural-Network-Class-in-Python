""" This module contains a neural network class

The module contains the neural network class used to implement multi-layer
neural networks.
"""
import random as rand
import tensorflow as tf
import numpy as np

class _ComputationalModel(object):
    """The purpose of this class is to build system of weights and
    activations that can return any necessary tensorflow objects.

    These objects are:
    - The input
    - The output
    - the dropout probabilities
    """
    def __init__(self, model_shape, activation_functions):
        """The user passes in the shape of the neural network and the
        activation functions for each level of the network.

        An example of what model_shape and activation_functions describes is
        if given [4,6,2] as model_shape then
        -a model with 4 input nodes, 6 hidden nodes, and 2 output nodes
        if given [sigmoid,sigmoid] as activation_functions then
        -a sigmoid function will be applied from the input nodes to
            the hidden nodes
        -a sigmoid function will be applied from the hidden nodes to the
            output nodes
        """
        self.input = tf.placeholder(tf.float32, [None, model_shape[0]], name='Input')
        self.filter = tf.placeholder(tf.float32, [None, model_shape[-1]], name='Mask')
        self.model_shape = model_shape
        self.dropout_prob_list = list()
        number_of_nodes_left = model_shape[0]
        i = 1
        self.activation_levels = [self.input]
        for number_of_nodes_right, activation_function in zip(model_shape[1:],
                                                              activation_functions):
            node = tf.Variable(tf.zeros([number_of_nodes_left,
                                         number_of_nodes_right]), name=('node_'+str(i)))
            probabiility_to_keep = tf.placeholder(tf.float32)
            node_with_dropout = tf.nn.dropout(node, probabiility_to_keep)
            bias = tf.Variable(tf.zeros([number_of_nodes_right]), name=('bias_'+str(i)))
            self.activation_levels.append(
                self.__choose_activation_function(activation_function,
                                                  node_with_dropout, bias))
            self.dropout_prob_list.append(probabiility_to_keep)
            number_of_nodes_left = number_of_nodes_right
            i += 1
        self.output = tf.mul(self.activation_levels[-1], self.filter, name="Output")

    def get_shape(self):
        """Returns the shape of computational model"""
        return self.model_shape

    def get_input(self):
        """Returns the input"""
        return self.input

    def get_output(self):
        """Returns the output"""
        return self.output

    def get_filter(self):
        """Returns the filter"""
        return self.filter

    def get_probabiliy_objects(self):
        """Returns a generator that returns an activation object"""
        return (activation for activation in self.dropout_prob_list)

    def __choose_activation_function(self, activation, drop, bia):
        """Used to choose which create the activation based on user input"""
        i = len(self.activation_levels) + 1
        if activation == "sigmoid":
            function = tf.sigmoid(tf.matmul(self.activation_levels[-1],
                                            drop)+bia, name=('sigmoid_'+str(i)))
        elif activation == "relu":
            function = tf.nn.relu(tf.matmul(self.activation_levels[-1],
                                            drop)+bia, name=('sigmoid_'+str(i)))
        elif activation == "linear":
            function = tf.matmul(self.activation_levels[-1], drop)+bia
        elif activation == "tanh":
            function = tf.tanh(tf.matmul(self.activation_levels[-1],
                                         drop)+bia, name=('sigmoid_'+str(i)))
        return function


class NeuralNet(object):
    """
    A class to implement multi-layer neural networks.

    This class allows the activation function, learning_rate, loss_function,
    number of layers, and number of hidden nodes for those layers.
    """
    def __init__(self, model_shape, learning_rate=0.1, activation_functions=None,
                 loss_function="squared_mean"):
        """
        Initializes the neural network with the parameters provided.

        level - a list of that describes the number of nodes for each layer
        learning_rate - the training learning rate of the neural network
        activations - a list of the activations for each layer (e.g. sigmoid)
        loss_function - the loss function to use ( can be a function or specify an
                        existing )
        """
        self.model = _ComputationalModel(model_shape, activation_functions)
        self.output_to_predict = tf.placeholder(tf.float32,
                                                [None, model_shape[-1]], name='prediction')
        if loss_function == "squared_mean":
            self.loss_function = tf.reduce_mean(tf.reduce_sum(tf.square(
                (self.model.get_output()-self.output_to_predict)
                ), reduction_indices=[1]))
        elif loss_function == "mean_absolute":
            self.loss_function = tf.reduce_mean(tf.reduce_sum(tf.abs(
                (self.model.get_output()-self.output_to_predict)
                ), reduction_indices=[1]))
        elif loss_function == "hinge_loss":
            self.loss_function = tf.reduce_mean(tf.reduce_sum(tf.abs(
                (1-self.model.get_output()*self.output_to_predict)
                ), reduction_indices=[1]))
        elif loss_function == "cross_entropy":
            self.loss_function = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.model.get_output(),
                    self.output_to_predict))
        self.train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(self.loss_function)
        #self.train_step = tf.train.AdadeltaOptimizer().minimize(self.lossFunction)
        #self.train_step = tf.train.RMSPropOptimizer(learningRate).minimize(self.lossFunction)
        #self.train_step = tf.train.AdamOptimizer().minimize(self.lossFunction)
        #self.train_step = tf.train.MomentumOptimizer(
        #    learning_rate, 0.6).minimize(self.loss_function)
        correct_prediction = tf.equal(self.output_to_predict, tf.round(
            self.model.get_output()))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), reduction_indices=[0])
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def training_run(self, nn_input, expected_output, **additional_params):
        """
        Train the neural network.

        nn_input - The input for the neural network
        expected_output - The expected output of neural network
        filt - *experimental* Used to ignore the error of certain outputs.
                If some sample lacks an expected output then ignoring the error
                will allow the sample to still be used despite this
        dropout_prob - A list of probabilities that determines the dropout rate
                       of a layer
        epochs - The number of epochs to train on

        Returns the loss of the model for every 10 percent of the epochs
        """
        dropout_probabilities = self.model.get_probabiliy_objects()
        additional_params["number of examples"] = np.shape(nn_input)[0]
        self.__setup_parameters(additional_params)
        dictionary = {self.model.get_input(): nn_input,
                      self.output_to_predict: expected_output,
                      self.model.get_filter(): additional_params["filter"]}
        for dropout_node, dropout_prob in zip(dropout_probabilities,
                                              additional_params["dropout_prob"]):
            dictionary[dropout_node] = dropout_prob
        to_return = []
        tenth = int(additional_params["epochs"]/10)
        indices = [range(len(nn_input))]
        for i in range(additional_params["epochs"]):
            rand.shuffle(indices)
            dictionary[self.model.get_input()] = nn_input[indices]
            dictionary[self.output_to_predict] = expected_output[indices]
            loss = self.sess.run([self.train_step,
                                  self.model.get_output(),
                                  self.loss_function,
                                  self.accuracy],
                                 feed_dict=dictionary)[2]
            if i%tenth == 0:
                to_return.append(loss)
        return to_return

    def run(self, nn_input, expected_output, **additional_params):
        """
        Runs the neural network with the model and returns the predicted output
        and the accuracy.

        nn_input - The input for the neural network
        expected_output - The expected output of the neural network
        filt - *experimental* Used to ignore the error of certain outputs.
                If some sample lacks an expected output then ignoring the error
                will allow the sample to still be used despite this
        dropout_prob - A list of probabilities that determines the dropout rate
                       of a layer
        """
        dropout_probabilities = self.model.get_probabiliy_objects()
        additional_params["number of examples"] = np.shape(nn_input)[0]
        self.__setup_parameters(additional_params)

        dictionary = {self.model.get_input(): nn_input,
                      self.model.get_filter(): additional_params["filter"],
                      self.output_to_predict: expected_output}
        for dropout_node, dropout_prob in zip(dropout_probabilities,
                                              additional_params["dropout_prob"]):
            dictionary[dropout_node] = dropout_prob
        # returns a prediction for the patient
        return self.sess.run([self.model.get_output(), self.accuracy], feed_dict=dictionary)

    def reset(self):
        """Resets the weights and bias."""
        self.sess.run(self.init)
        #self.train_step = tf.train.GradientDescentOptimizer(learningRate).
        #minimize(self.lossFunction)
        #self.train_step = tf.train.AdadeltaOptimizer().minimize(self.lossFunction)
        #self.train_step = tf.train.RMSPropOptimizer(learningRate).minimize(self.lossFunction)
        #self.train_step = tf.train.AdamOptimizer().minimize(self.lossFunction)

    def __setup_parameters(self, parameters):
        shape = self.model.get_shape()
        if "dropout_prob" not in parameters:
            parameters["dropout_prob"] = [1]*len(shape)
        if "epochs" not in parameters:
            parameters["epochs"] = 1
        if "filter" not in parameters:
            if shape[-1] == 1:
                parameters["filter"] = np.array([[1]*parameters["number of examples"]]).transpose()
            else:
                parameters["filter"] = [[1]*shape[-1]]*parameters["number of examples"]
