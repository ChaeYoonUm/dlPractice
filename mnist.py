import tensorflow as tf
# import tensorflow_addons as tfa
# import tqdm
#import ipykernel
#import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import datetime

#data load
# 60000개의 트레이닝 데이터
# 10000개의 테스트 데이터
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling 0~255 to 0~1
x_train, x_test = x_train / 255.0, x_test / 255.0


#===CNN Model===#
model = tf.keras.models.Sequential([
    #filter: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    #input_shape=(28,28,1): 1->흑백채널
    # filter 값은 kernel_initializer에 의해 초기에는 아주 작은 랜덤 값으로 채워짐
    tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=3, filters=16, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    
])
#learning rate decay
# step_decay = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96, staircase=True)

#train 시작전 모델 구성 및 compile
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model 구조 출력 
model.summary()

#tensorboard 보기
#tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#model train
print("Fit model")

#log 저장 -> tensorboard에서 출력
# history = model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=50,
#                     validation_data=(x_test, y_test))
history = model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=32,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback])

#model 평가
print("Evaluate model")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

predicted_result = model.predict(x_test)
print(f'predicted result: {predicted_result.shape}')
predicted_labels = np.argmax(predicted_result,  axis=1) #가로방향
print(f'predicted labels: {predicted_labels[:10]}')


#틀린 데이터 모으기
wrong_result = []
for n in range(0, len(y_test)):
    if predicted_labels[n] != y_test[n]:
        wrong_result.append(n)
print(f'wrong data: {len(wrong_result)}')

#틀린 데이터 출력_16개만 랜덤으로 뽑아서
import random
samples = random.choices(population=wrong_result, k =16)
plt.figure(figsize=(14, 12))
for idx, n in enumerate(samples):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(x_test[n].reshape(28,28), cmap = 'Greys', interpolation='nearest')
    plt.title('Label ' + str(y_test[n]) + ', Predict ' + str(predicted_labels[n]))
    plt.axis('off')    
#plt.show()

