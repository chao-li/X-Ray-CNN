from keras import optimizers
from keras.callbacks import ModelCheckpoint
from models import BaselineNet
from models import ShallowNet
from models import MicroVGGNet
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
from models.callbacks import TrainingMonitor
import os

#FILE LOCATIONS
model_name = 'BaselineNet_ADAM_Epoch30'
# data location
data_folder = '/home/ubuntu/image_as_numpy/'
# output path
output_path = '/home/ubuntu/X-Ray-CNN/outputs'
monitor_path = '/home/ubuntu/X-Ray-CNN/monitor'


# load the model
#model = ShallowNet.build(width = 128, height = 128, depth = 1, classes = 1, dense_size = 2000)
model = BaselineNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
#model = MicroVGGNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 1e-4),
	metrics = ['accuracy'])

model.summary()


# load the data
from keras.preprocessing.image import ImageDataGenerator

# load the train and validation data
X_train = np.load(data_folder + 'X_train.npy')
X_validate = np.load(data_folder + 'X_validate.npy')
y_train = np.load(data_folder + 'y_train.npy')
y_validate = np.load(data_folder + 'y_validate.npy')


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

    
validate_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(X_train)
validate_datagen.fit(X_validate)

train_generator = train_datagen.flow(X_train, y_train, batch_size = 32)
validation_generator = validate_datagen.flow(X_validate, y_validate, batch_size = 32)


## SAVING OUTPUTS AND LOGS
# create the image callback
figPath = os.path.sep.join([monitor_path, model_name + '.png'])
jsonPath = os.path.sep.join([monitor_path, model_name + '.json'])

# create checkpoint
checkpoint = ModelCheckpoint(output_path + '/' +  model_name + '|_best_weights.hdf5', monitor = 'val_acc',mode = 'max', save_best_only = True, verbose = 1)
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), checkpoint]

# TRAINING THE MODEL
history = model.fit_generator(train_generator,
                                  steps_per_epoch = len(X_train)/32, # 264 batches per epoch\n",
                                  epochs = 30,
                                  validation_data = validation_generator,
                                  validation_steps = len(X_validate)/32,
                                  callbacks = callbacks)

model.save(output_path + '/' + model_name +  '|_final_result.hdf5')
