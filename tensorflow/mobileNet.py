import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import datetime

#dataset

#===MobileNet===#
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(224,224,3), kernel_size=3, filters=32)
    
])
#MobileNet opt => RMSprop
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop)

model.summary()