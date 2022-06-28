# @Author: shounak.ray
# @Date:   2022-06-28T09:44:02-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-28T11:19:21-07:00

import matplotlib.pyplot as plt
import math
import numpy as np
import itertools


def _soft_sanitation(variable, msg='Cannot complete operation; requires previous step.'):
    if variable is None:
        print(msg)
        return


class SOM:
    def __init__(self, neurons, learning_rate, epochs, **kwargs):
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._node_min_value = kwargs.get('NODE_MIN_VALUE', 0)
        self._node_max_value = kwargs.get('NODE_MAX_VALUE', 100)
        self._weight_min = kwargs.get('NODE_WEIGHT_MIN', 0)
        self._weight_max = kwargs.get('NODE_WEIGHT_MAX', 1)

        self.curr_epoch = None

    def create_feature_map(self, num_features):
        self.num_features = num_features
        _single_dim_length = math.ceil(np.sqrt(self.neurons))
        self.adj_neurons = _single_dim_length * _single_dim_length
        print(f"> Initializing {self.adj_neurons} neurons in the feature map...")
        _node_stepsize = (self._node_max_value - self._node_min_value) / _single_dim_length
        _all_r, _all_c = np.array([np.arange(self._node_min_value, self._node_max_value, _node_stepsize)] * 2)

        def _random_weight_vector():
            return np.array([np.random.uniform(self._weight_min, self._weight_max) for _ in range(num_features)])

        def _get_all_neurons(_all_r, _all_c):
            return np.array([Neuron(_random_weight_vector(), coordinate)
                             for coordinate in itertools.product(_all_r, _all_c)])

        self.neuronal_data = _get_all_neurons(_all_r, _all_c)
        print("> Feature map initialized.\n")

    def plot_neurons(self):
        _soft_sanitation(self.neuronal_data, msg='Create the feature map first before trying to plot the neurons.')
        plt.figure(figsize=(15, 15))
        plt.title(f"Feature map on epoch = {self.curr_epoch} of {self.epochs}")
        plt.scatter(*zip(*[n.coordinates for n in self.neuronal_data]))
        plt.show()


class Neuron:
    def __init__(self, weight_vector, coordinates):
        self.weight_vector = weight_vector
        self.coordinates = coordinates


S = SOM(100, 1, 1)
S.create_feature_map(8)
S.plot_neurons()

# EOF
