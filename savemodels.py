# %%
from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# %%
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1, 28 * 28) / 255
test_images = test_images[:10000].reshape(-1, 28 * 28) / 255

# %%


def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model


model = create_model()
model.summary()

# %%
checkpoint_path = 'training/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model = create_model()
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# %%
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print('Untrained model, accuracy: {:5.2f}%'.format(100 * acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# %%
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, period=5
)

model = create_model()
model.fit(
    train_images,
    train_labels,
    epochs=50,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0
)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# %%
model = create_model()
model.load_weights(latest))
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# %%
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %%
model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %%
model = create_model()

model.fit(train_images, train_labels, epochs=5)

saved_model_path = tf.contrib.saved_model.save_keras_model(
    model, "./saved_models")
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model

# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
