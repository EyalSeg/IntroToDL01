import pytest
import numpy as np
import math

from sys import float_info

from nn.dense_layer import DenseLayer
from nn_test.gradient_test import GradientTest, assert_decays
from verification import jacobian_test


class TestDenselayer(GradientTest):

    @pytest.fixture()
    def net(self):
        return DenseLayer()

    @pytest.fixture()
    def weights(self, X, c):
        return np.random.random((X.shape[0], c.shape[0]))

    @pytest.fixture()
    def biases(self, c):
        return np.random.random(c.shape[0])

    def test_jacMV_wrt_biases(self, net, X, weights, biases, c):
        f = lambda b: net.forward(X[:, 0], weights, b)
        jacMV = lambda b, v: net.jacobian_biases(X[:, 0], weights, b) @ v.reshape(-1)
        quadratic, linear = jacobian_test(f, jacMV, biases, verbose=True,
                                          plot=True, title='Jacobian Test w.r.t. biases')

        assert_decays(quadratic, linear)

    def test_jacMV_wrt_weights(self, net, X, weights, biases, c):
        f = lambda w: net.forward(X[:, 0], w, biases)
        jacMV = lambda w, v: net.jacobian_weights(X[:, 0], w, biases) @ v.reshape(-1)
        quadratic, linear = jacobian_test(f, jacMV, weights, verbose=True,
                                          plot=True, title='Jacobian Test w.r.t. weights')

        assert_decays(quadratic, linear)

    def test_jacMV_wrt_data(self, net, X, weights, biases, c):
        f = lambda x: net.forward(x, weights, biases)
        jacMV = lambda x, v: net.jacobian_data(x, weights, biases) @ v.reshape(-1)
        quadratic, linear = jacobian_test(f, jacMV, X[:, 0], verbose=True,
                                          plot=True, title='Jacobian Test w.r.t. data')

        assert_decays(quadratic, linear)

    def test_jacTMV_wrt_biases(self, net, X, weights, biases, c):
        v = np.random.random(c.shape[0])[:, np.newaxis]
        u = np.random.random(c.shape[0])[:, np.newaxis]

        x = X[:, 0]
        x_batched = x[:, np.newaxis]

        jacMV = net.jacobian_biases(x, weights, biases) @ v.reshape(-1)
        jacTMU = net.jacTMV_biases(x_batched, weights, biases, u)

        ut_jacMV = u.transpose() @ jacMV
        vt_jacTMU = v.transpose() @ jacTMU

        assert math.fabs(np.sum(ut_jacMV - vt_jacTMU)) <= float_info.epsilon * 2

    def test_jacTMV_wrt_weights(self, net, X, weights, biases, c):
        v = np.random.random(weights.shape)
        u = np.random.random(c.shape[0])[:, np.newaxis]

        x = X[:, 0]
        x_batched = x[:, np.newaxis]

        jacMV = net.jacobian_weights(x, weights, biases) @ v.transpose().reshape(-1)
        jacTMU = net.jacTMV_weights(x_batched, weights, biases, u).transpose().reshape(-1)[:, np.newaxis]

        ut_jacMV = u.transpose() @ jacMV
        vt_jacTMU = v.transpose().reshape(-1) @ jacTMU

        diff = math.fabs(np.sum(ut_jacMV - vt_jacTMU))
        assert diff <= float_info.epsilon * 2

    def test_jacTMV_wrt_data(self, net, X, weights, biases, c):
        x = X[:, 0]
        x_batched = x[:, np.newaxis]

        v = np.random.random(x_batched.shape)
        u = np.random.random(c.shape[0])[:, np.newaxis]

        jacMV = net.jacobian_data(x, weights, biases) @ v.reshape(-1)
        jacTMU = net.jacTMV_data(x_batched, weights, biases, u)

        ut_jacMV = u.transpose() @ jacMV
        vt_jacTMU = v.transpose() @ jacTMU

        assert math.fabs(np.sum(ut_jacMV - vt_jacTMU)) <= float_info.epsilon * 2
