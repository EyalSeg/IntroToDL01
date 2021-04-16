import pytest
import numpy as np

from sys import float_info

from softmax import Softmax
from nn_test.gradient_test import assert_decays, GradientTest
from verification import gradient_test

allowed_error = 0.5


class TestSoftmax(GradientTest):

    @pytest.fixture()
    def weights(self, softmax):
        return softmax.get_random_weights()

    @pytest.fixture()
    def biases(self, softmax):
        return softmax.get_random_biases()

    @pytest.fixture()
    def softmax(self, X, c):
        return Softmax(X.shape[0], c.shape[0])

    def test_gradient_wrt_weights(self, softmax,  X, weights, biases, c):
        f = lambda W: softmax.loss(X, W, biases, c)
        f_gradient = lambda W: softmax.gradient_wrt_weights(X, W, biases, c)
        quadratic, linear = gradient_test(f, f_gradient, weights, verbose=True,
                                          plot=True, title='Gradient Test w.r.t. weights')

        assert_decays(quadratic, linear)

    def test_gradient_wrt_biases(self, softmax, X, weights, biases, c):
        f = lambda b: softmax.loss(X, weights, b, c)
        f_gradient = lambda b: softmax.gradient_wrt_biases(X, weights, b, c)
        quadratic, linear = gradient_test(f, f_gradient, biases, verbose=True,
                                          plot=True, title='Gradient Test w.r.t. biases')

        assert_decays(quadratic, linear)

    def test_gradient_wrt_data(self, softmax, X, weights, biases, c):
        f = lambda x: softmax.loss(x, weights, biases, c)
        f_gradient = lambda x: softmax.gradient_wrt_data(x, weights, biases, c)
        quadratic, linear = gradient_test(f, f_gradient, X, verbose=True,
                                          plot=True, title='Gradient Test w.r.t. data')

        assert_decays(quadratic, linear)

    def test_output_shape(self, softmax, X, weights, biases, c):
        output = softmax.forward(X, weights, biases)
        assert output.shape == c.shape

    def test_output_scale(self, softmax, X, weights, biases, c):
        output = softmax.forward(X, weights, biases)
        sums = np.array([np.sum(output[:, i]) for i in range(output.shape[1])])

        assert np.all(sums >= 1 - float_info.epsilon * 2)
        assert np.all(sums <= 1 + float_info.epsilon * 2)
