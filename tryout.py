import tensorflow as tf
import numpy as np

w = tf.Variable(tf.random.normal((3, 2)), name="w")
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name="b")
x = [[1.0, 2.0, 3.0]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)
    print(type(loss))
    print(loss)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])

print(w.shape)
print(dl_dw)
