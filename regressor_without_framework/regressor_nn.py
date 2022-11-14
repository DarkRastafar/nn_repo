import numpy as np


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class Gradient:
    def __init__(self):
        self.w1 = 1.2
        self.b = 15
        self.epoch = 10000

    def f(self, x):
        return self.w1 * x + self.b

    def stop_by_loss(self, y_true, x_pred, eps):
        # print(((y_true - x_pred) ** 2).mean())
        # if np.sum((y_true - x_pred) ** 2) < eps:
        #     return True

        if ((y_true - x_pred) ** 2).mean() < eps:
            return True

    # def update_weights_MSE(self, m, b, X, Y, learning_rate):
    #     m_deriv = 0
    #     b_deriv = 0
    #     N = len(X)
    #     for i in range(N):
    #         # -2x(y - (mx + b))
    #         m_deriv += -2 * X[i] * (Y[i] - (m * X[i] + b))
    #         # -2(y - (mx + b))
    #         b_deriv += -2 * (Y[i] - (m * X[i] + b))
    #         # Мы вычитаем, потому что производная указывает на самый крутой подъем
    #
    #         m -= (m_deriv / float(N)) * learning_rate
    #         b -= (b_deriv / float(N)) * learning_rate

    def gradient_descent(self, x_train, y_test, learning_rate=0.001, eps=0.1):
        history = []
        history_pred = []
        N = len(x_train)
        for i in range(self.epoch):
            for x, y_true in zip(x_train, y_test):
                x_pred = self.f(x)
                history_pred.append(x_pred)

                if self.stop_by_loss(y_true, x_pred, eps):
                    break
                else:
                    # print(f'epoch {i} --> x === {x} {round(learning_rate * (y_true - x_pred), 4)}')
                    # print(f'epoch {i} --> x === {x} {round(learning_rate * x * (y_true - x_pred), 4)}')
                    # print(self.w1)
                    # print(self.b)
                    # print(f'x_pred ---> {x_pred}')
                    # print(f'y_true ---> {y_true}')
                    # print(f'loss ---> {np.sum((y_true - x_pred) ** -2)}')
                    # print()

                    # self.w1 += round(learning_rate * (y_true - x_pred), 4)
                    # self.b += round(learning_rate * x * (y_true - x_pred), 4)

                    m_deriv = -2 * x * (y_true - x_pred)
                    b_deriv = -2 * (y_true - x_pred)
                    self.w1 -= (m_deriv / float(N)) * learning_rate
                    self.b -= (b_deriv / float(N)) * learning_rate

            if i % 10 == 0:
                y_preds = np.apply_along_axis(self.f, 0, x_train)
                loss = mse_loss(y_test, y_preds)
                history.append(loss)
                print(i, loss)

        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.grid(True)
        plt.show()
        #
        # plt.plot(history_pred)
        # plt.grid(True)
        # plt.show()


if __name__ == '__main__':
    x_train = np.array([-40, -10, 0, 8, 15, 22, 38])
    y_test = np.array([-40, 14, 32, 46, 59, 72, 100])

    nn_gr = Gradient()
    nn_gr.gradient_descent(x_train, y_test)
    # print()
    # print(nn_gr.w1)
    # print(nn_gr.b)
    # print()
    print(nn_gr.f(8))
    print(nn_gr.f(38))
    print(nn_gr.f(22))
    print(nn_gr.f(20))
