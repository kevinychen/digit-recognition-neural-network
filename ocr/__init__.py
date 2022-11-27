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
mult = .001
train = 100000
batch = 100

weights = [np.random.normal(size=(shape[lvl + 1], shape[lvl])) for lvl in range(num_levels)]
biases = [np.zeros(shape[lvl + 1]) for lvl in range(num_levels)]
for loop in range(5):
    for n in range(0, train, batch):
        neurons = [np.array([np.vectorize(lambda x: x / 256)(image.flatten()) for image in images[n:n+batch]])]
        for lvl in range(num_levels):
            neurons.append(np.vectorize(sigma)(neurons[lvl] @ weights[lvl].T) + biases[lvl])

        expected = [[1 if i == label - 1 else 0 for i in range(shape[-1])] for label in labels[n:n+batch]]
        dE_da = neurons[-1] - expected
        for lvl in range(num_levels)[::-1]:
            dE_dv = dE_da * np.vectorize(dsigma)(neurons[lvl + 1])
            dE_da = dE_dv @ weights[lvl]
            weights[lvl] -= mult * dE_dv.T @ neurons[lvl]
            biases[lvl] -= mult * np.sum(dE_dv, axis=0)

        error = np.sum((neurons[-1] - expected) ** 2)
        if n % 10000 == 0:
            print(n, error)

test = []
for n in range(train, train + 10000):
    neurons = [np.vectorize(lambda x: x / 256)(images[n].flatten())]
    for lvl in range(num_levels):
        neurons.append(np.vectorize(sigma)(weights[lvl] @ neurons[lvl]) + biases[lvl])
    test.append(np.argmax(neurons[-1]) == labels[n] - 1)
print(sum(test), len(test))
