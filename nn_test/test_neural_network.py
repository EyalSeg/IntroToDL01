import pytest
import numpy as np

from nn.neural_network import NeuralNetwork
from nn_test.gradient_test import GradientTest, assert_decays
from verification import gradient_test


class TestNeuralNetwork(GradientTest):
    @pytest.fixture()
    def hidden_dims(self):
        return [4, 5, 4]

    @pytest.fixture()
    def net(self, X, c, hidden_dims):
        dims = [X.shape[0]] + hidden_dims + [c.shape[0]]
        return NeuralNetwork(dims)

    @pytest.fixture()
    def W(self, net):
        return net.get_random_weights()

    @pytest.fixture()
    def B(self, net):
        return net.get_random_biases()

    def test_weights_vectorization(self, net, W):
        v = net.vectorise(W)
        W2 = net.devectorise_weights(v)

        for w1, w2 in zip(W, W2):
            assert np.array_equal(w1, w2)

    def test_biases_vectorization(self, net, B):
        v = net.vectorise(B)
        B2 = net.devectorise_biases(v)

        for b1, b2 in zip(B, B2):
            assert np.array_equal(b1, b2)

    def test_output_shape(self, net, X, c, W, B):
        output = net.forward(X, W, B)
        assert output.shape == c.shape, "output of forward pass is of wrong shape!"

    def test_gradient_weights(self, net, X, c, W, B):
        f = lambda w: net.loss(X, w, B, c)
        f_vectorised = lambda wv: f(net.devectorise_weights(wv))

        f_gradient = lambda w: net.gradient_wrt_weights(X, w, B, c)
        f_gradient_vectorised = lambda wv: net.vectorise(f_gradient(net.devectorise_weights(wv)))

        quadratic, linear = gradient_test(f_vectorised,
                                          f_gradient_vectorised,
                                          net.vectorise(W),
                                          verbose=True,
                                          plot=True,
                                          one_graph=True,
                                          title='Gradient Test of all Network w.r.t. weights'
                                          )

        assert_decays(quadratic, linear)

    def test_gradient_biases(self, net, X, c, W, B):
        f = lambda b: net.loss(X, W, b, c)
        f_vectorised = lambda bv: f(net.devectorise_biases(bv))

        f_gradient = lambda b: net.gradient_wrt_biases(X, W, b, c)
        f_gradient_vectorised = lambda bv: net.vectorise(f_gradient(net.devectorise_biases(bv)))

        quadratic, linear = gradient_test(f_vectorised,
                                          f_gradient_vectorised,
                                          net.vectorise(B),
                                          verbose=True,
                                          plot=True,
                                          one_graph=True,
                                          title='Gradient Test of all Network w.r.t. biases'
                                          )

        assert_decays(quadratic, linear)

