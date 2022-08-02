import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

tmp1 = x_train
tmp2 = x_test
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

cnn = models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

rnn = models.Sequential([
    layers.SimpleRNN(32, input_shape=(28, 28), activation='relu', return_sequences=True),
    layers.SimpleRNN(32, input_shape=(28, 28), activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_history = cnn.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)
x_train, x_test = tmp1, tmp2
rnn_history = rnn.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)

plt.plot(cnn_history.history['accuracy'], label='cnn_acc')
plt.plot(cnn_history.history['val_accuracy'], label='cnn_val')
plt.plot(rnn_history.history['accuracy'], label='rnn_acc')
plt.plot(rnn_history.history['val_accuracy'], label='rnn_val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
