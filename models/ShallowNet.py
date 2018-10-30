# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

class ShallowNet:
	@staticmethod
	def build(width, height, depth, output, dense_size):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# define the first (and only) CONV => RELU layer
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))

		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(MaxPooling2D(pool_size = (2,2)))

		# softmax classifier
		model.add(Flatten())

		model.add(Dense(dense_size, activation = 'relu'))

		model.add(Dense(output))
		
	
		model.add(Activation('sigmoid'))


		# return the constructed network architecture
		return model
