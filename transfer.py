from tensorflow.keras import datasets, layers, models, applications, optimizers

base_model = applications.DenseNet121(input_shape=(32, 32, 3),
                                      include_top=False,      # remove last layer
                                      weights='imagenet')

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

base_model.trainable = False

for layer in base_model.layers[-50:]:
    layer.trainable = True  # unfreeze

new_model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

adam_low_rate = optimizers.Adam(learning_rate=0.0005)

new_model.compile(optimizer=adam_low_rate,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )


history = new_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)
test_loss, test_acc = new_model.evaluate(x_train, y_train)

print(test_loss, test_acc)