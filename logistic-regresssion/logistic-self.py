import numpy as np
from sklearn.datasets import load_iris


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


class LogisticRegressionLearner:

    def __init__(self, x):
        self.x = x
        self.w = np.random.normal(size=(x.shape[1], 1))
        print(f"logistic learner created with input shape {x.shape} and weight shape {self.w.shape}")

    def predict(self):
        return sigmoid(np.dot(self.x, self.w))

    def cost(self, labels):
        cost = -labels * np.log(self.predict()) - (1 - labels) * np.log(1 - self.predict())
        cost = cost.sum() / len(labels)
        return cost


def main():
    x, y = load_iris(return_X_y=True)

    train_start_index = 0
    train_end_index = int(x.shape[0] * 0.8)
    x_train = x[train_start_index:train_end_index, :]
    y_train = y[train_start_index:train_end_index]
    y_train = np.reshape(y_train, (-1, 1))
    x_valid = x[train_end_index:, :]
    y_valid = y[train_end_index:]
    logistic_learner = LogisticRegressionLearner(x_train)

    epochs = 1000
    N = x.shape[0]
    lr = 0.001
    for epoch in range(epochs):
        prediction = logistic_learner.predict()
        grad = np.dot(x_train.T, (prediction - y_train))
        grad /= N
        grad *= lr
        logistic_learner.w -= grad
        if epoch % 100 == 0:
            cost = logistic_learner.cost(y_train)
            print(f"cost after epoch {epoch} is {cost}")


if __name__ == '__main__':
    main()
