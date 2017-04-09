import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout
import visualiser

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
input_length = 3072
num_encoders = 1000


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reshape
X_train = X_train[:5]
X_train_flattened = X_train.reshape(len(X_train), input_length)[:5]
# X_test= X_test.reshape(len(X_test), 3072)[:3]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32') / 255.0
X_train_flattened = X_train_flattened.astype('float32') / 255.0


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(758, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(input_length, activation='sigmoid'))

# Compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, X_train_flattened, epochs=epochs, batch_size=100)
# Final evaluation of the model
# scores = model.evaluate(X_test, X_train_flattened, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

visualiser.visualise_keras(model, X_train, dim_x=32, dim_y=32, extra_dim=3)