import numpy as np


class DenseLayer:
    def __init__(self, activation='tanh'):
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        else:
            raise Exception(f"Undefined activation: {activation}")

    def forward(self, x, w, b):
        linear_products = np.array([x.transpose() @ w[:, i] + b[i] for i in range(w.shape[1])])
        activated = self.activation(linear_products)

        return activated

    def jacobian_biases(self, x, w, b):
        linear_products = w.transpose() @ x + b
        derivative_activated = self.activation_derivative(linear_products)
        jacobian = np.diag(derivative_activated)

        return jacobian

    def jacTMV_biases(self, x, w, b, v):
        linear_products = w.transpose() @ x + b[:, np.newaxis]

        derivative_activated = self.activation_derivative(linear_products)

        return derivative_activated * v

    def jacTMV_weights(self, x, w, b, v):
        return self.jacTMV_biases(x, w, b, v) @ x.transpose()

    def jacTMV_data(self, x, w, b, v):
        return w @ self.jacTMV_biases(x, w, b, v)

    def jacobian_weights(self, x, w, b):
        jac = self.jacobian_biases(x, w, b)
        kronecker = np.kron(x.transpose(), np.eye(w.shape[1]))

        jacobian = jac @ kronecker
        return jacobian

    def jacobian_data(self, x, w, b):
        jac = self.jacobian_biases(x, w, b)

        return jac @ w.transpose()







