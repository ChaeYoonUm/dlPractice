import tensorflow as tf
import numpy as np
import keras

test_list1 = [1, 2, 3]
test_list2 = [10, 20, 30]
initializer = tf.constant_initializer(value=0.5)

t1 = tf.Variable(test_list1, dtype=tf.float32)
t2 = tf.Variable(test_list2, dtype=tf.float32)
w1 = tf.Variable(initializer(shape=[1000,100], dtype=tf.float32))
w2 = tf.Variable(initializer(shape=[100,1000], dtype=tf.float32))

with tf.GradientTape() as tape:
    t3 = t1 * t2

optimizer = tf.keras.optimizers.Adam (1e-5)
gradients = tape.gradient(t3, [t1, t2])

print(gradients[0])
print(gradients[1])
