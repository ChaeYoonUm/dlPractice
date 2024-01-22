import tensorflow as tf
# import tensorflow_addons as tfa
# import tqdm

#import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

#data load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#scaling 0~255 to 0~1
x_train, x_test = x_train / 255.0, x_test / 255.0

#===Dense Model===#
"""
# train 데이터의 첫번째 이미지를 grayscale 로 표시
#plt.imshow(x_train[0], cmap='gray')
#plt.show()
# train 데이터의 첫번째 라벨 프린트 
#print(y_train[0])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax'),
])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

#model.summary()
predictions = model(x_train[:1]).numpy()
#tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
#print(loss_fn(y_train[:1], predictions).numpy()) => 2.33482 출력 -> 초기 loss -log(1/10) 값으로 잘 나옴
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
"""

#===CNN Model: AlexNet===#
model = tf.keras.models.Sequential([
    #filter: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    #input_shape=(28,28,1): 1->흑백채널
    # filter 값은 kernel_initializer에 의해 초기에는 아주 작은 랜덤 값으로 채워짐
    #
    # tf.keras.layers.Conv2D(input_shape=(28,28,1), activation = 'relu', kernel_size=3, filters=32, padding='same'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D((2,2)),
    
    # tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D((2,2)),
    
    # tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=3, filters=32, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    
])
#train 시작전 모델 구성 및 compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#model 구조 출력 
model.summary()

#model train
print("Fit model")
#tqdm_callback = tfa.callbacks.TQDMProgressBar()
history = model.fit(x_train, y_train, epochs=10, verbose=1,
                    validation_data=(x_test, y_test))

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
    
plt.show()
