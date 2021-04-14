import numpy as np
import math


def gradient_test(f, f_grad, x, epsilon_range=10, epsilon_0=1, verbose=False):
    d = np.random.random(x.shape)
    d /= np.sum(d)

    linearily_decreasing = []
    quadratically_decreasing = []

    epsilons = [0.5 ** i * epsilon_0 for i in range(epsilon_range)]

    if verbose:
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

    return quadratic_decay, linear_decay


def jacobian_test(f, jacMV, x, epsilon_range=10, epsilon_0=1, verbose=False):
    d = np.random.random(x.shape)
    d /= np.sum(d)

    linearily_decreasing = []
    quadratically_decreasing = []

    epsilons = [0.5 ** i * epsilon_0 for i in range(epsilon_range)]

    if verbose:
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

    return quadratic_decay, linear_decay


if __name__ == "__main__":
    f = lambda x: x ** 2
    f_grad = lambda x: 2 * x

    gradient_test(f, f_grad, np.array([5]))