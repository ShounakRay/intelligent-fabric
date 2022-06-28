# @Author: shounak.ray
# @Date:   2022-06-28T09:44:02-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-28T10:42:21-07:00

import matplotlib.pyplot as plt
import math
import numpy as np
import itertools


class SOM:
    def __init__(neurons, learning_rate, epochs):
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.epochs = epochs


class Neuron:
    def __init__(self, weight_vector, coordinates):
        self.weight_vector = weight_vector
        self.coordinates = coordinates


n = 144
NODE_MIN_VALUE, NODE_MAX_VALUE = 0, 100
_single_dim_length = math.ceil(np.sqrt(n))
n_adjusted = _single_dim_length * _single_dim_length
_node_stepsize = (NODE_MAX_VALUE - NODE_MIN_VALUE) / _single_dim_length
all_r, all_c = [np.arange(NODE_MIN_VALUE, NODE_MAX_VALUE, _node_stepsize)] * 2

WEIGHT_MIN, WEIGHT_MAX = 0, 1


def _random_weight_vector(num_features, WEIGHT_MIN, WEIGHT_MAX):
    return np.array([np.random.uniform(WEIGHT_MIN, WEIGHT_MAX) for _ in range(num_features)])


def _get_all_neurons(all_r, all_c):
    return np.array([Neuron(_random_weight_vector(10), coordinate) for coordinate in itertools.product(all_r, all_c)])


ALL_NEURONS = _get_all_neurons(all_r, all_c)

# EOF
