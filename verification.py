import numpy as np
import math
import matplotlib.pyplot as plt


def gradient_test(f, f_grad, x, epsilon_range=10, epsilon_0=1, verbose=False, plot=False, title='',
                  one_graph=True):
    d = np.random.random(x.shape)
    d /= np.sum(d)

    linearily_decreasing = []
    quadratically_decreasing = []

    epsilons = [0.5 ** i * epsilon_0 for i in range(epsilon_range)]

    if verbose:
        print('\n' + title)
        print("Should decrease quadratically\tShould decrease linearily\t\tratio\n")

    for epsilon in epsilons:
        should_decrease_linearily = math.fabs(
            f(x + epsilon * d)
            -
            f(x))

        should_decrease_quadratically = math.fabs(
            f(x + epsilon * d)
            -
            f(x)
            -
            np.matmul((epsilon * d.reshape(-1)),
                      f_grad(x).reshape(-1))
        )

        if should_decrease_linearily == 0 or should_decrease_quadratically == 0:
            break

        linearily_decreasing.append(should_decrease_linearily)
        quadratically_decreasing.append(should_decrease_quadratically)

        if verbose:
            print(f"{should_decrease_quadratically}\t\t\t\t{should_decrease_linearily}\t\t\t\t"
                f"{should_decrease_quadratically / should_decrease_linearily}")

    linear_ratios = [a/b for a, b in zip(linearily_decreasing[:-1], linearily_decreasing[1:])]
    quadratic_ratios = [a/b for a, b in zip(quadratically_decreasing[:-1], quadratically_decreasing[1:])]

    quadratic_decay = sum(quadratic_ratios) / len(quadratic_ratios)
    linear_decay = sum(linear_ratios) / len(linear_ratios)

    if verbose:
        print(f"quadratic rations: {quadratic_ratios}")
        print(f"linear rations: {linear_ratios}")

        print(f"quadratic decay: {quadratic_decay}\t\t\t(should be ~4)")
        print(f"linear decay: {linear_decay}\t\t\t(should be ~2)")

    if (plot):
        indices = range(len(linearily_decreasing))
        plt.plot(indices, linearily_decreasing)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Delta")

        if not one_graph:
            plt.legend(["First Taylor Order"])
            plt.show()

        plt.plot(indices, quadratically_decreasing)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Delta")

        if not one_graph:
            plt.legend(["Second Taylor Order"])
        else:
            plt.legend(["First Taylor Order", "Second Taylor Order"])

        plt.show()

    return quadratic_decay, linear_decay


def jacobian_test(f, jacMV, x, epsilon_range=10, epsilon_0=1, verbose=False, plot=False, title='',
                  one_graph=True):
    d = np.random.random(x.shape)
    d /= np.sum(d)

    linearily_decreasing = []
    quadratically_decreasing = []

    epsilons = [0.5 ** i * epsilon_0 for i in range(epsilon_range)]

    if verbose:
        print('\n' + title)
        print("Should decrease quadratically\tShould decrease linearily\t\tratio\n")

    for epsilon in epsilons:
        should_decrease_linearily = math.fabs(np.sum(
            f(x + epsilon * d)
            -
            f(x)))

        should_decrease_quadratically = math.fabs(np.sum(
            f(x + epsilon * d)
            -
            f(x)
            -
            jacMV(x, epsilon * d).transpose()
        ))

        if should_decrease_linearily == 0 or should_decrease_quadratically == 0:
            break

        linearily_decreasing.append(should_decrease_linearily)
        quadratically_decreasing.append(should_decrease_quadratically)

        if verbose:
            print(f"{should_decrease_quadratically}\t\t\t\t{should_decrease_linearily}\t\t\t\t"
                f"{should_decrease_quadratically / should_decrease_linearily}")

    linear_ratios = [a/b for a, b in zip(linearily_decreasing[:-1], linearily_decreasing[1:])]
    quadratic_ratios = [a/b for a, b in zip(quadratically_decreasing[:-1], quadratically_decreasing[1:])]

    quadratic_decay = sum(quadratic_ratios) / len(quadratic_ratios)
    linear_decay = sum(linear_ratios) / len(linear_ratios)

    if verbose:
        print(f"quadratic rations: {quadratic_ratios}")
        print(f"linear rations: {linear_ratios}")

        print(f"quadratic decay: {quadratic_decay}\t\t\t(should be ~4)")
        print(f"linear decay: {linear_decay}\t\t\t(should be ~2)")

    if (plot):
        indices = range(len(linearily_decreasing))
        plt.plot(indices, linearily_decreasing)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Delta")

        if not one_graph:
            plt.legend(["First Taylor Order"])
            plt.show()

        plt.plot(indices, quadratically_decreasing)
        plt.title(title)

        if not one_graph:
            plt.legend(["Second Taylor Order"])
        else:
            plt.legend(["First Taylor Order", "Second Taylor Order"])

        plt.show()

    return quadratic_decay, linear_decay