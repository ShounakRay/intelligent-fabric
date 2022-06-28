# @Author: shounak.ray
# @Date:   2022-06-28T09:44:02-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-28T13:25:26-07:00

import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
import networkx as nx
import pandas as pd
from sklearn import datasets


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
        self._node_max_value = kwargs.get('NODE_MAX_VALUE', 1)
        self._weight_min = kwargs.get('NODE_WEIGHT_MIN', 0)
        self._weight_max = kwargs.get('NODE_WEIGHT_MAX', 1)

        self.curr_epoch = None

    def create_feature_map(self, num_features):
        def naivetuple_to_pos(naive_tuple, _single_num_neurons, _node_stepsize):
            return (naive_tuple[0] * _node_stepsize, naive_tuple[1] * _node_stepsize)

        def _random_weight_vector(num_features):
            return np.array([np.random.uniform(self._weight_min, self._weight_max) for _ in range(num_features)])

        # Adjust mapping dimensions, if required
        self.num_features = num_features
        _single_num_neurons = math.ceil(np.sqrt(self.neurons))
        self.adj_neurons = _single_num_neurons * _single_num_neurons
        print(f"> Initializing {self.adj_neurons} neurons in the feature map...")
        _node_stepsize = (self._node_max_value - self._node_min_value) / _single_num_neurons

        # Finally make the mapping
        G = nx.grid_2d_graph(_single_num_neurons, _single_num_neurons)
        attrs = {node: {'type': 'neuron',
                        'position': naivetuple_to_pos(node, _single_num_neurons, _node_stepsize),
                        'weight_vector': _random_weight_vector(self.num_features)} for node in G.nodes()}
        nx.set_node_attributes(G, attrs)
        self.neuronal_data = G
        print("> Feature map initialized.\n")

    def plot_neurons(self):
        plt.figure(figsize=(15, 15))
        nx.draw(self.neuronal_data, pos=nx.get_node_attributes(self.neuronal_data, 'position'),
                node_color='lightgreen',
                with_labels=True,
                node_size=600)

    def fit(self, data):
        if len(data[0]) != self.num_features:
            print("FATAL: The number of features detected in the data doesn't match what was entered during map creation. Redo map creation.")
            return
        if len(data) == 0:
            print("FATAL: You entered an empty dataset. Retry.")
            return
        input_vector = data[np.random.randint(0, len(data))]
        bmu_index = np.argmin([np.linalg.norm(x - input_vector)
                               for x in list(nx.get_node_attributes(S.neuronal_data, 'weight_vector').values())], axis=0)
        bmu = np.array(S.neuronal_data.nodes)[bmu_index]


data_1 = pd.DataFrame(datasets.load_iris()['data'], columns=datasets.load_iris()[
    'feature_names'])
data = data_1.to_numpy()

S = SOM(100, 1, 1)
S.create_feature_map(4)
S.plot_neurons()
list(S.neuronal_data.neighbors((3, 3)))

# EOF
