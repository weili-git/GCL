import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = models.Sequential([
    layers.LSTM(32, input_shape=(28, 28), activation='relu', return_sequences=True),
    layers.LSTM(32, input_shape=(28, 28), activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)
test_loss, test_acc = model.evaluate(x_train, y_train, batch_size=64)

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

