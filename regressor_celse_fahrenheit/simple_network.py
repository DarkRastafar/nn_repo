import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping


c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])


def run_celse():
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.1, restore_best_weights=False)
    single_feature_normalizer = keras.layers.Normalization(input_shape=[1, ], axis=None)

    model = keras.Sequential([
        single_feature_normalizer,
        Dense(units=1, activation='linear')
    ])
    model.compile(loss=keras.losses.LogCosh(), optimizer=keras.optimizers.Adam(0.15))
    history = model.fit(c, f, epochs=100, verbose=0, callbacks=[early_stop])

    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.show()

    print(model.predict([8]))
    print(model.predict([-10]))
    print(model.predict([0]))
    print(model.predict([15]))
    print(model.predict([22]))
    print(model.predict([38]))


if __name__ == '__main__':
    run_celse()
