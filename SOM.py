# @Author: shounak.ray
# @Date:   2022-06-28T09:44:02-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-29T19:17:14-07:00

import string
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
import networkx as nx
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
import sys
from scipy.spatial.distance import squareform, cdist
from sklearn.manifold import MDS  # for MDS dimensionality reduction
import seaborn as sns
import scipy
# from celluloid import Camera
from datetime import datetime
from tqdm import tqdm
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation
import imageio.v2 as imageio
import glob
import os


def _soft_sanitation(variable, msg='Cannot complete operation; requires previous step.'):
    if variable is None:
        print(msg)
        return


class SOM:
    def __init__(self, neurons, learning_rate, epochs, sigma_0, convergence_threshold, **kwargs):
        self.neurons = float(neurons)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.tau_rate = float(kwargs.get('tau_rate', epochs))
        self.sigma_0 = float(sigma_0)
        self.tau_neighbourhood = float(kwargs.get('tau_rate', epochs / np.log(sigma_0)))
        self.convergence_threshold = float(convergence_threshold)

        self._node_min_value = kwargs.get('NODE_MIN_VALUE', 0)
        self._node_max_value = kwargs.get('NODE_MAX_VALUE', 1)
        self._weight_min = kwargs.get('NODE_WEIGHT_MIN', 0)
        self._weight_max = kwargs.get('NODE_WEIGHT_MAX', 1)

        self.adjustment_history = []
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
                        'weight_vector': _random_weight_vector(self.num_features),
                        'adjustment_history': []} for node in G.nodes()}
        nx.set_node_attributes(G, attrs)
        self.neuronal_data = G
        print("> Feature map initialized.\n")

    def plot_neurons(self, figsize=(10, 10), literal=False, only_draw_nodes=False, **kwargs):
        # if kwargs.get('ax') is not None:
        #     kwargs['ax'].clear()
        hist = {k: v for k, v in nx.get_node_attributes(self.neuronal_data, 'adjustment_history').items() if v != []}
        colors = ['green' if k in list(hist.keys()) else 'red' for k in self.neuronal_data.nodes]

        _viridis = cm.get_cmap('viridis', 8)
        colors_by_weight = np.array([np.linalg.norm(wv)
                                    for wv in nx.get_node_attributes(self.neuronal_data, 'weight_vector')])
        colors_by_weight = MinMaxScaler(feature_range=(0, 1)).fit_transform(colors_by_weight.reshape(-1, 1))
        colors_by_weight = [_viridis(x) for x in colors_by_weight]

        sizes = np.array([np.linalg.norm(self.neuronal_data[node]) for node in self.neuronal_data.nodes])
        sizes = MinMaxScaler(feature_range=(200, 500)).fit_transform(sizes.reshape(-1, 1))

        position = nx.get_node_attributes(self.neuronal_data, 'weight_vector' if literal else 'position')

        kwargs = {'color': colors, 'size': sizes, 'color_weight': colors_by_weight, 'position': position}
        # if kwargs.get('being_animated', False) is False:
        #     plt.figure(figsize=figsize)
        _ = plt.figure(figsize=(10, 10))
        if only_draw_nodes:
            img = nx.draw_networkx_nodes(self.neuronal_data, pos=kwargs.get('position', nx.spring_layout(self.neuronal_data)),
                                         node_color=kwargs.get('color_weight', 'lightgreen'),
                                         node_size=kwargs.get('sizes', 200),
                                         ax=kwargs.get('ax'))
        else:
            img = nx.draw(self.neuronal_data, pos=kwargs.get('position', nx.spring_layout(self.neuronal_data)),
                          node_color=kwargs.get('color_weight', 'lightgreen'),
                          with_labels=False,
                          node_size=kwargs.get('sizes', 200),
                          ax=kwargs.get('ax'))
        plt.savefig(kwargs.get('fname', f'pictures/neurons-{self.curr_epoch}.png'))
        plt.close()

    def _plot_new_layer(self, **kwargs):
        coordinates = list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())
        model2d = MDS(n_components=2,
                      metric=True,
                      n_init=4,
                      max_iter=300,
                      verbose=0,
                      eps=0.001,
                      n_jobs=-1,
                      random_state=42,
                      dissimilarity='euclidean')
        X_trans = model2d.fit_transform(coordinates)
        # plt.figure(figsize=(8, 8))
        _ = plt.figure(figsize=(10, 10))
        if (ax := kwargs.get('ax')) is not None:
            pl.title(f'Epoch {self.curr_epoch}')
        _ = plt.scatter(x=X_trans[:, 0], y=X_trans[:, 1], alpha=0.9, c='black', s=2)
        plt.savefig(kwargs.get('fname', f'pictures/neurons-{self.curr_epoch}.png'))
        plt.close()
        # plt.show()

    def fit(self, data, animate=True, anim_every_n_epochs=10, literal=False, **kwargs):
        print('> Fitting map...')
        if len(data[0]) != self.num_features:
            print("FATAL: The number of features detected in the data doesn't match what was entered during map creation. Redo map creation.")
            return
        if len(data) == 0:
            print("FATAL: You entered an empty dataset. Retry.")
            return

        def _get_neighbouring_nodes(bmu):
            return [bmu] + list(nx.neighbors(self.neuronal_data, bmu))

        def _get_random_input_vector(data):
            return data[np.random.randint(0, len(data) - 1)]

        def _get_bmu(input_vector):
            bmu_index = np.argmin([np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())],
                                  axis=0)
            # plt.hist([np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())], bins=30)
            # [np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())][bmu_index]
            return list(self.neuronal_data.nodes)[bmu_index]  # BMU

        if animate:
            # Delete existing files in folder
            [os.remove(f) for f in glob.glob('pictures/*')]
            # make folder if it doesn't exist
            os.makedirs(r'pictures') if not os.path.exists(r'pictures') else None
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        try:
            for epoch in tqdm(range(1, self.epochs + 1)):
                self.curr_epoch = epoch
                input_vector = _get_random_input_vector(data)
                bmu = _get_bmu(input_vector)
                if animate and epoch % anim_every_n_epochs == 0 and epoch < self.epochs:
                    self.plot_neurons(literal=True, only_draw_nodes=kwargs.get('only_draw_nodes')) if literal else self._plot_new_layer(**kwargs)
                    # camera.snap()
                adj_mags = [self.update_weight(epoch, bmu, neighbour, input_vector) for neighbour in _get_neighbouring_nodes(bmu)]
                self.adjustment_history.append(mean_change := np.mean(adj_mags))
                if mean_change <= self.convergence_threshold:
                    print(f"Early stopping at epoch {epoch}. Convergence threshold reached.")
                    break
        except KeyboardInterrupt as e:
            print('Interrupted...working with what we have.')

        if animate:
            nums = sorted([int(''.join(filter(str.isdigit, s))) for s in glob.glob('pictures/*.png')])
            img_paths = [f'pictures/neurons-{n}.png' for n in nums]
            ims = [imageio.imread(f) for f in img_paths]
            imageio.mimwrite('file.mp4', ims, fps=60)
            plt.close()
        else:
            self.plot_neurons(literal=True, only_draw_nodes=kwargs.get('only_draw_nodes')) if literal else self._plot_new_layer(**kwargs)
            plt.show()

        print('> Done fitting map.')
        # self._plot_new_layer(ax)

    def update_weight(self, epoch, bmu, neighbour, input_vector):
        def adaptive_eta(epoch):
            return self.learning_rate * math.exp(-epoch / self.tau_rate)
            # return self.learning_rate / (1 + epoch / (self.epochs / 2))  # ALT

        def adaptive_sigma(epoch):
            return self.sigma_0 * math.exp(-epoch / self.tau_neighbourhood)

        def topological_neighourhood(epoch, neighbour, bmu):
            # Range (0, 1)
            lateral_distance = np.linalg.norm(self.neuronal_data.nodes[bmu]['weight_vector'] - self.neuronal_data.nodes[neighbour]['weight_vector'])
            # sigma = constrain(self.sigma_0 / (1 + epoch / self.epochs))  # ALT
            return math.exp(-(lateral_distance**2) / (2 * adaptive_sigma(epoch)**2))

        # def constrain(value):
        #     # https://stackoverflow.com/questions/1835787/what-is-the-range-of-values-a-float-can-have-in-python
        #     # if value >= sys.float_info.max * sys.float_info.epsilon:
        #     #     return sys.float_info.max * sys.float_info.epsilon
        #     # if value <= sys.float_info.min * sys.float_info.epsilon:
        #     #     return sys.float_info.min * sys.float_info.epsilon
        #     return value

        current_weight = self.neuronal_data.nodes[neighbour]['weight_vector']
        adjustment = adaptive_eta(epoch) * topological_neighourhood(epoch, neighbour, bmu) * (input_vector - current_weight)
        self.neuronal_data.nodes[neighbour]['weight_vector'] = current_weight + adjustment
        magnitude = np.sqrt(adjustment.dot(adjustment))  # adjustment magnitude
        self.neuronal_data.nodes[neighbour]['adjustment_history'].append(magnitude)
        return magnitude
        # print(f"Cosine Similarity: {1 - scipy.spatial.distance.cosine(current_weight, current_weight + adjustment)}")


""" SKLEARN """
# data = pd.DataFrame(datasets.load_iris()['data'], columns=datasets.load_iris()['feature_names']).to_numpy()
# data = pd.DataFrame(datasets.fetch_covtype()['data'],
#                     columns=datasets.fetch_covtype()['feature_names']).to_numpy()
""" ARTIFICAL CLUSTER – 3 BLOBS """
# data, labels_true = make_blobs(n_samples=1000, centers=[[0, 0], [2.5, 4], [5, 0]], cluster_std=0.8, random_state=0)
# data = (data - data.min(0)) / data.ptp(0)
# # _ = plt.scatter(*zip(*data))

""" ARTIFICAL CLUSTER – SMILY FACE """
data, labels_true = make_blobs(n_samples=5000, centers=[[0, 4], [4, 4]], cluster_std=0.2, random_state=0)
data_arc = np.array([[i, -0.5 * np.sin(i)] for i in np.arange(0, np.pi, 0.1)])
data_arc = [make_blobs(n_samples=80, centers=[c], cluster_std=0.1, random_state=0)[0] for c in data_arc]
data_arc = np.concatenate(data_arc)
data = np.vstack((data, data_arc))
data = np.hstack((data, np.random.normal(0, 1, len(data)).reshape(-1, 1)))
# data = np.hstack((data, np.random.normal(0, 1, len(data)).reshape(-1, 1)))
data = (data - data.min(0)) / data.ptp(0)
# _ = plt.scatter(*zip(*data))


def run_model():
    neurons = 5 * np.sqrt(len(data))
    learning_rate = 0.25
    epochs = 50000
    sigma_0 = 20
    convergence_threshold = 1e-3

    S = SOM(neurons=neurons, learning_rate=learning_rate, epochs=epochs, sigma_0=sigma_0, convergence_threshold=convergence_threshold)
    S.create_feature_map(len(data[0]))
    S.fit(data, animate=False, literal=False, anim_every_n_epochs=20000, only_draw_nodes=True)
    S._plot_new_layer(fname='final_layering.png')

# _ = plt.plot(S.adjustment_history[100000:150000])


if __name__ == '__main__':
    run_model()

# EOF
