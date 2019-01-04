# %%
import tensorflow as tf
tf.enable_eager_execution()

# %%
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("Hello World"))

print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# %%


def time_matmul(x):
    %timeit tf.matmul(x, x)


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
