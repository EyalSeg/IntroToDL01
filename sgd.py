import numpy as np

from sklearn.utils import shuffle


class SGD:
    def __init__(self, objective, learn_rate=0.0006, minibatch_size=10):
        self.objective = objective
        self.learn_rate = learn_rate
        self.minibatch_size = minibatch_size

    def fit(self, X, c, w, b, vectorise=False, epochs=10, epoch_callbacks=()):
        if vectorise:
            wv = self.objective.vectorise(w)
            bv = self.objective.vectorise(b)

        for callback in epoch_callbacks:
            callback(0, w, b)

        for epoch in range(epochs):
            X_batches, c_batches = self.__batchify(X, c)

            for X_batch, c_batch in zip(X_batches, c_batches):
                _ = self.objective.forward(X_batch, w, b) # gradients in dnn require a forward pass
                grad_w = self.objective.gradient_wrt_weights(X_batch, w, b, c_batch)
                grad_b = self.objective.gradient_wrt_biases(X_batch, w, b, c_batch)

                if vectorise:
                    grad_w = self.objective.vectorise(grad_w)
                    grad_b = self.objective.vectorise(grad_b)
                    wv -= grad_w * self.learn_rate
                    bv -= grad_b * self.learn_rate

                    w = self.objective.devectorise_weights(wv)
                    b = self.objective.devectorise_biases(bv)

                else:
                    w -= grad_w * self.learn_rate
                    b -= grad_b * self.learn_rate

            for callback in epoch_callbacks:
                callback(0 + epoch, w, b)

        return w, b

    def __batchify(self, X, c):
        X, c = self.__shuffle(X, c)
        n_batches = np.ceil(X.shape[1] / self.minibatch_size)

        X_batches = np.array_split(X, n_batches, axis=1)
        c_batches = np.array_split(c, n_batches, axis=1)

        return X_batches, c_batches

    @staticmethod
    def __shuffle(X, c):
        X, c = shuffle(X.transpose(), c.transpose())

        return X.transpose(), c.transpose()


if __name__ == '__main__':
    from data.data_loader import DataLoader
    from softmax import Softmax

    data = DataLoader.load_dataset("PeaksData.mat")

    X_train = data['Yt']
    c_train = data['Ct']

    X_validate = data['Yv']
    c_validate = data['Cv']

    objective = Softmax()

    #callback = lambda weights: print(objective.loss(X_validate, weights, c_validate))

    sgd = SGD(objective)
    weights = sgd.fit(X_train, c_train, epochs=100)

    print(objective.loss(X_validate, weights, c_validate))



