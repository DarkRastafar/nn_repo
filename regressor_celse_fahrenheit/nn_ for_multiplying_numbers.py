import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from tensorflow import keras
from keras.callbacks import EarlyStopping


X = np.array([
    [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10],
    [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10],
    [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10],
    [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10],
    [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10],
    [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10],
    [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10],
    [1, 2], [3, 1], [3, 9]
])


Y = np.array([
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    6, 12, 18, 24, 30, 36, 42, 48, 54, 60,
    7, 14, 21, 28, 35, 42, 49, 56, 63, 70,
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80,
    9, 18, 27, 36, 45, 54, 63, 72, 81, 90,
    2, 3, 27
])


def run_network():
    early_stop = EarlyStopping(monitor='loss', patience=100, verbose=1, min_delta=0.1, restore_best_weights=False)

    single_feature_normalizer = keras.layers.Normalization(input_shape=[2, ], axis=None)
    model = keras.Sequential([
        single_feature_normalizer,
        Dense(units=1, activation='linear')
    ])
    model.compile(loss=keras.losses.LogCosh(), optimizer=keras.optimizers.Adam(0.15))
    history = model.fit(X, Y, epochs=100, verbose=0, callbacks=[early_stop])

    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.show()

    print(model.predict([2, 5]))
    print(model.predict([4, 4]))
    print(model.predict([5, 5]))
    print(model.predict([3, 9]))
    print(model.predict([5, 8]))
    print(model.predict([9, 9]))


if __name__ == '__main__':
    run_network()
