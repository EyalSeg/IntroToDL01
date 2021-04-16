import optuna
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from data.data_loader import DataLoader
from nn.neural_network import NeuralNetwork
from sgd import SGD
from softmax import Softmax

sns.set_theme(style="darkgrid")

def tune(experiment, n_trials, n_jobs):
    objective = lambda trial: experiment(trial.suggest_float("learn_rate", 0.0000001, 0.01))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"best parameters are: {study.best_params} with a value of {study.best_value}")


def measure_accuracy(objective, X, w, b, c):
    predictions = objective.forward(X, w, b)
    predictions = [np.argmax(predictions[:, i]) for i in range(predictions.shape[1])]

    ground_truth = np.array([np.argmax(ci) for ci in c.transpose()])

    errors = np.count_nonzero(ground_truth - predictions)

    return 1 - (errors / X.shape[1])

def run_experiment(**kwargs):
    data = kwargs['data']

    X_train = data['Yt']
    c_train = data['Ct']

    X_validate = data['Yv']
    c_validate = data['Cv']

    objective = kwargs['objective']

    w = objective.get_random_weights()
    b = objective.get_random_biases()

    optimizer = SGD(objective, learn_rate=kwargs['learn_rate'], minibatch_size=kwargs['batch_size'])

    if kwargs['plot']:
        plt.figure()

        result_y_train = []
        result_y_validate = []

        result_y_train_loss = []
        result_y_validate_loss = []

        measure_train = lambda index, weights, biases: \
            result_y_train.append(measure_accuracy(objective, X_train, weights, biases, c_train))
        measure_validate = lambda index, weights, biases: \
            result_y_validate.append(measure_accuracy(objective, X_validate, weights, biases, c_validate))

        measure_train_loss = lambda index, weights, biases: \
            result_y_train_loss.append(objective.loss(X_train, weights, biases, c_train))
        measure_validate_loss = lambda index, weights, biases: \
            result_y_validate_loss.append(objective.loss(X_validate, weights, biases, c_validate))

        callbacks = [measure_train, measure_validate, measure_train_loss, measure_validate_loss]
    else:
        callbacks = []

    w, b = optimizer.fit(X_train,
                  c_train,
                  w,
                  b,
                  epochs=kwargs['epochs'],
                  epoch_callbacks=callbacks,
                  vectorise=kwargs['vectorise'])

    if kwargs['plot']:
        df = pd.DataFrame.from_dict({"training set": result_y_train,
                                     "validation set": result_y_validate})
        df.index.name = "Epoch"

        sns.lineplot(data=df, dashes=False)
        lrn_str = "{:12.7f}".format(kwargs['learn_rate'])
        plt.title(f"{kwargs['title']}\n learning rate = {lrn_str}")
        plt.ylabel("Accuracy")
        plt.show()

        df_loss = pd.DataFrame.from_dict({"training set": result_y_train_loss,
                                     "validation set": result_y_validate_loss})
        df_loss.index.name = "Epoch"
        sns.lineplot(data=df_loss, dashes=False)
        lrn_str = "{:12.7f}".format(kwargs['learn_rate'])
        plt.title(f"{kwargs['title']}\n learning rate = {lrn_str}")
        plt.ylabel("Loss")

        plt.show()

    return objective.loss(X_validate, w, b, c_validate)


def run_softmax_experiment(dataset_name, learn_rate, batch_size, epochs, plot):
    data = DataLoader.load_dataset(dataset_name)

    X_train = data['Yt']
    c_train = data['Ct']

    softmax = Softmax(X_train.shape[0], c_train.shape[0])
    return run_experiment(title=dataset_name + " softmax",
                   data=data,
                   objective=softmax,
                   epochs=epochs,
                   learn_rate=learn_rate,
                   vectorise=False,
                   batch_size=batch_size,
                   plot=plot
                   )


def run_nn_experiment(dataset_name, hidden_dims, learn_rate, batch_size, epochs, plot):
    data = DataLoader.load_dataset(dataset_name)

    X_train = data['Yt']
    c_train = data['Ct']

    dims = [X_train.shape[0]] + hidden_dims + [c_train.shape[0]]
    net = NeuralNetwork(dims)

    return run_experiment(title=dataset_name + " neural network",
                   data=data,
                   objective=net,
                   epochs=epochs,
                   learn_rate=learn_rate,
                   vectorise=True,
                   batch_size=batch_size,
                   plot=plot
                   )


if __name__ == "__main__":
    batch_size = 100
    hidden_dims = [50, 50, 50, 50, 50]
    epochs = 150

    tune_parameters = False

    if (tune_parameters):
        tune_obj = lambda lrn_rate: \
            run_nn_experiment("PeaksData.mat", hidden_dims=hidden_dims, learn_rate=lrn_rate, epochs=epochs,
                              batch_size=batch_size, plot=False)
        tune(tune_obj, n_trials=100, n_jobs=4)

    nn_exp = True

    print("NN exp: " + str(nn_exp) +
        "\nBatch size: " + str(batch_size) +
          "\nHidden layers dimensions: " + str(hidden_dims) +
          "\nEpochs: " + str(epochs))

    if not nn_exp:
        run_softmax_experiment("PeaksData.mat", learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)
        run_softmax_experiment("GMMData.mat", learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)
        run_softmax_experiment("SwissRollData.mat", learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)

    else:
        run_nn_experiment("PeaksData.mat", hidden_dims=hidden_dims, learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)
        run_nn_experiment("GMMData.mat", hidden_dims=hidden_dims, learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)
        run_nn_experiment("SwissRollData.mat", hidden_dims=hidden_dims, learn_rate=0.008594211858637247, epochs=epochs, batch_size=batch_size, plot=True)


