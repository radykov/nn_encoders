import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from scipy.misc import toimage
import visualiser

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train_original, y_train_original), (X_test_original, y_test_orignal) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train_original.shape[1] * X_train_original.shape[2]
X_train = X_train_original.reshape(X_train_original.shape[0], num_pixels).astype('float32')
X_test = X_test_original.reshape(X_test_original.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
# y_train = np_utils.to_categorical(X_train)
# y_test = np_utils.to_categorical(X_test)
# num_classes = y_test.shape[1]

num_encoders = 50

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_encoders, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(num_encoders, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_pixels, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()

# model.predict(X_train)
# Fit the model
model.fit(X_train, X_train, validation_data=(X_train, X_train), epochs=2, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, X_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

visualiser.visualise_keras(model, X_test, transform_required=True)