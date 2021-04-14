import numpy as np

from data.data_loader import DataLoader
from xavier_initialization import init

class Softmax:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_random_weights(self):
        return init((self.input_shape, self.output_shape))

    def get_random_biases(self):
        return init(self.output_shape)

    def forward(self, X, w, b):
        score_exps, total = self.__get_score_exponents(X, w, b)
        softmax = score_exps / total

        return softmax

    def loss(self, X, w, b, c):
        scores, total_scores = self.__get_score_exponents(X, w, b)

        label_scales = [score / total_scores for score in scores]
        label_losses = [ci.transpose() @ np.log(scale) for scale, ci in zip(label_scales, c)]

        loss = np.sum(label_losses) / -X.shape[1]
        return loss

    def gradient_wrt_weights(self, X, w, b, c):
        n_labels = w.shape[1]

        score_exps, total_score_exp = self.__get_score_exponents(X, w, b)

        inners = [score_exps[lbl] / total_score_exp - c[lbl] for lbl in range(n_labels)]
        gradient = np.array([(X @ inner) / X.shape[1] for inner in inners]).transpose()

        return gradient

    def gradient_wrt_data(self, X, w, b, c):
        n_labels = w.shape[1]
        m = X.shape[1]

        scores = [w[:, label_index].transpose() @ X + b[label_index] for label_index in range(n_labels)]
        # etta = np.max(scores)

        temp = w.transpose() @ X
        for i in range(temp.shape[1]):
            temp[:, i] += b

        scores_exp = np.exp(temp)
        scores_exp_sum = sum([np.exp(score) for score in scores])

        inner = scores_exp / scores_exp_sum - c

        gradient = w @ inner / m

        return gradient

    def gradient_wrt_biases(self, X, w, b, c):
        n_labels = w.shape[1]

        score_exps, total_score_exp = self.__get_score_exponents(X, w, b)

        inners = [score_exps[lbl] / total_score_exp - c[lbl] for lbl in range(n_labels)]
        gradient = np.array([np.sum(inner) / X.shape[1] for inner in inners])

        return gradient

    def __get_score_exponents(self, X, w, b):
        n_labels = w.shape[1]

        scores = [self.__label_sum(X, w, b, lbl) for lbl in range(n_labels)]
        etta = np.max(scores)
        score_exp = np.array([np.exp(score - etta) for score in scores])
        total_scores = sum(score_exp)

        return score_exp, total_scores

    def __label_sum(self, X, w, b, label_index):
        return X.transpose() @ w[:, label_index] + b[label_index]