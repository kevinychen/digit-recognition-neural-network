from emnist import *
import math
import numpy as np


def sigma(x):
    if x > 100:
        return 1
    elif x < -100:
        return 0
    else:
        return 1 / (1 + math.exp(-x))


def dsigma(x):
    if abs(x) > 100:
        return 0
    return sigma(x) * (1 - sigma(x))


images, labels = extract_training_samples('digits')

shape = [784] + [20] * 2 + [10]
num_levels = len(shape) - 1
mult = .01
train = 100000

weights = [np.random.normal(size=(shape[lvl + 1], shape[lvl])) for lvl in range(num_levels)]
biases = [np.zeros(shape[lvl + 1]) for lvl in range(num_levels)]
for n in range(train):
    neurons = [np.vectorize(lambda x: x / 256)(images[n].flatten())]
    for lvl in range(num_levels):
        neurons.append(np.vectorize(sigma)(np.matmul(weights[lvl], neurons[lvl]) + biases[lvl]))

    expected = [1 if i == labels[n] - 1 else 0 for i in range(shape[-1])]
    dE_da = 2 * (neurons[-1] - expected)
    for lvl in range(num_levels, 0, -1):
        for i in range(len(neurons[lvl])):
            da_dw = dsigma(neurons[lvl][i]) * neurons[lvl - 1]
            weights[lvl - 1][i] -= mult * dE_da[i] * da_dw
        da_db = np.vectorize(dsigma)(neurons[lvl])
        biases[lvl - 1] -= mult * dE_da * da_db
        dE_da = np.matmul(np.transpose(weights[lvl - 1]), dE_da * np.vectorize(dsigma)(neurons[lvl]))

    error = sum((neurons[-1] - expected) ** 2)
    if n % 1000 == 0:
        print(n, error)

test = []
for n in range(train, train + 10000):
    neurons = [np.vectorize(lambda x: x / 256)(images[n].flatten())]
    for lvl in range(num_levels):
        neurons.append(np.vectorize(sigma)(np.matmul(weights[lvl], neurons[lvl]) + biases[lvl]))

    test.append(np.argmax(neurons[-1]) == labels[n] - 1)
print(sum(test), len(test))
