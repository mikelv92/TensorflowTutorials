# %%
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print(tf.__version__)

# %%
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Train set: {}".format(train_data.shape))
print("Train labels: {}".format(train_labels.shape))
print("Test set: {}".format(test_data.shape))
print("Test labels: {}".format(test_labels.shape))

# %%
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# %%
# Normalize features
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# %%
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu,
                                 input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer=tf.train.RMSPropOptimizer(
        0.001), metrics=['mae'])
    return model


model = build_model()
model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print(' ')
        print('.', end='')


epochs = 500
history = model.fit(train_data, train_labels, epochs=epochs,
                    validation_split=0.2, verbose=0, callbacks=[PrintDot()])


# %%
def plot_history(history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Validation loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)


# %%
model = build_model()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=0,
                    callbacks=[early_stopping, PrintDot()])
plot_history(history)

# %%
(loss, mae) = model.evaluate(test_data, test_labels, verbose=0)
print('Testing set mean absolute error: ${:7.2f}'.format(mae * 1000))

# %%
test_predictions = model.predict(test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# %%
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel('Prediction error')
plt.ylabel('Count')
plt.show()
