import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
trainX = train_x.reshape((train_x.shape[0], 28, 28 , 1))
testX = test_x.reshape((test_x.shape[0], 28, 28 , 1))
trainY = tf.keras.utils.to_categorical(train_y)
testY = tf.keras.utils.to_categorical(test_y)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY))

signiture = infer_signature(testX, model.predict(testX))
mlflow.keras.log_model(model, "mnist_cnn", signature=signiture)