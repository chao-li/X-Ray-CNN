
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class BaselineNet:
        @staticmethod
        def build(width, height, depth, classes, dense_size):
                # initialize the model along with the input shape to be
                # "channels last" and the channels dimension itself
                model = Sequential()
                inputShape = (height, width, depth)
                chanDim = -1

                # if we are using "channels first", update the input shape
                # and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1

                # first CONV => RELU => CONV => RELU => POOL layer set
                model.add(Conv2D(32, (3, 3), activation = 'relu', padding = "same",
                        input_shape=inputShape))
                model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Conv2D(64, (3, 3), activation = 'relu', padding="same"))
                model.add(MaxPooling2D(pool_size=(2, 2)))


                # second CONV => RELU => CONV => RELU => POOL layer set
                model.add(Conv2D(128, (3, 3), activation = 'relu', padding="same"))
                model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Conv2D(128, (3, 3), activation = 'relu', padding="same"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # first (and only) set of FC => RELU layers
                model.add(Flatten())

                model.add(Dense(dense_size, activation = 'relu'))
                model.add(Dropout(0.5))

                # first (and only) set of FC => RELU layers
                model.add(Dense(classes))

                if classes < 2:
                        model.add(Activation('sigmoid'))
                else:
                        model.add(Activation('softmax'))

                # return the constructed network architecture
                return model
