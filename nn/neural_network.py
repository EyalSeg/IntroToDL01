import numpy as np

from nn.dense_layer import DenseLayer
from softmax import Softmax
from xavier_initialization import init


class NeuralNetwork:
    def __init__(self, dims):
        dim_pairs = list(zip(dims[0:-1], dims[1:]))

        self.layers = [DenseLayer() for _ in dim_pairs[:-1]]
        self.softmax = Softmax(dims[-2], dims[-1])
        self.weight_shapes = dim_pairs
        self.bias_shapes = dims[1:]
        self.x_cache = []

    def get_random_weights(self):
        return [init(shape) for shape in self.weight_shapes]

    def get_random_biases(self):
        return [init(shape) for shape in self.bias_shapes]

    def vectorise(self, W):
        return np.concatenate([w.reshape(-1) for w in W])

    def devectorise_weights(self, Wv):
        sizes = [np.prod(shape) for shape in self.weight_shapes]
        indices = [0]

        for size_idx in range(len(sizes) - 1):
            indices.append(indices[-1] + sizes[size_idx])

        buffers = [Wv[idx: idx + size] for idx, size in zip(indices, sizes)]
        weights = [buffer.reshape(shape) for buffer, shape in zip(buffers, self.weight_shapes)]

        return weights

    def devectorise_biases(self, Bv):
        sizes = [shape[1] for shape in self.weight_shapes]
        indices = [0]

        for size_idx in range(len(sizes) - 1):
            indices.append(indices[-1] + sizes[size_idx])

        biases = [Bv[idx: idx + size] for idx, size in zip(indices, sizes)]
        return biases

    def loss(self, X, W, B, c):
        x = self.__forward_hidden(X, W, B)
        return self.softmax.loss(x, W[-1], B[-1], c)

    def forward(self, X, W, B):
        x = self.__forward_hidden(X, W, B)
        return self.softmax.forward(x, W[-1], B[-1])

    def gradient_wrt_weights(self, X, W, B, c):
        gradient = []

        grad = self.softmax.gradient_wrt_weights(self.x_cache[-1], W[-1], B[-1], c)
        gradient.append(grad)
        v = self.softmax.gradient_wrt_data(self.x_cache[-1], W[-1], B[-1], c)

        for layer_i in reversed(range(len(self.layers))):
            layer, x, w, b = self.layers[layer_i], self.x_cache[layer_i], W[layer_i], B[layer_i]
            grad = layer.jacTMV_weights(x, w, b, v)
            gradient.append(grad.transpose())

            v = layer.jacTMV_data(x, w, b, v)

        return list(reversed(gradient))

    def gradient_wrt_biases(self, X, W, B, c):
        gradient = []

        grad = self.softmax.gradient_wrt_biases(self.x_cache[-1], W[-1], B[-1], c)
        gradient.append(grad.transpose())
        v = self.softmax.gradient_wrt_data(self.x_cache[-1], W[-1], B[-1], c)

        for layer_i in reversed(range(len(self.layers))):
            layer, x, w, b = self.layers[layer_i], self.x_cache[layer_i], W[layer_i], B[layer_i]
            grad = sum(layer.jacTMV_biases(x, w, b, v).transpose())[:, np.newaxis]
            gradient.append(grad.transpose())

            v = layer.jacTMV_data(x, w, b, v)

        return list(reversed(gradient))

    def __forward_hidden(self, x, W, B):
        self.x_cache = []

        for layer, w, b in zip(self.layers, W, B):
            self.x_cache.append(x)
            x = layer.forward(x, w, b)

        self.x_cache.append(x)
        return x


