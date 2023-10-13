import numpy as np


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)
        self.input = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weight.T) + self.bias

    def backward(self, grad_output):
        self.grad_weight = np.dot(grad_output.T, self.input)
        self.grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weight)
        return grad_input

class MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / len(y_true)


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layer):
        layer.weight -= self.learning_rate * layer.grad_weight
        layer.bias -= self.learning_rate * layer.grad_bias


class LinearReg:
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        loss_fn = 'mse',
        optimizer = 'sgd',
        lr = 1e-3,
        epochs = 30,
        interval_verbose = None
    ) -> None:
        loss_map = {
            'mse': MSE
        }
        opt_map = {
            'sgd': SGD
        }

        self.linear = LinearLayer(in_feature, out_feature)
        self.loss_fn = loss_map[loss_fn]()
        self.opt = opt_map[optimizer](learning_rate=lr)
        self.epochs = epochs
        self.interval_verbose = interval_verbose

    def fit(self, x, y):
        for epoch in range(self.epochs):
            y_pred = self.linear.forward(x)

            loss = self.loss_fn.forward(y_pred, y)
            grad_loss = self.loss_fn.backward(y_pred, y)
            grad_input = self.linear.backward(grad_loss)

            self.opt.step(self.linear)

            if self.interval_verbose is not None \
                and epoch % self.interval_verbose == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        return self.linear.forward(x)
