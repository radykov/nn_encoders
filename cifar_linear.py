import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
import visualiser

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reshape
X_train = X_train.reshape(len(X_train), 3072)
X_test= X_test.reshape(len(X_test), 3072)[:3]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

num_encoders = 1000

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_encoders, input_dim=3072, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3072, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# model.predict(X_train)
# Fit the model
model.fit(X_train, X_train, epochs=10, batch_size=100, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, X_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

visualiser.visualise_keras(model, X_test, dim_x=32, dim_y=32, extra_dim=3, transform_required=True)