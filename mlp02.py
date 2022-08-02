import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='SAME'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=64)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


