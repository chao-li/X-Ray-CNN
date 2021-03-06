from keras import optimizers
from keras.callbacks import ModelCheckpoint
from models import BaselineNet
from models import ShallowNet
from models import MicroVGGNet
from models import BaselineNet_NoPad
from models import BaselineNet_LeakyRelu
from models import AveragePoolingNet
from keras.callbacks import LearningRateScheduler
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
from models.callbacks import TrainingMonitor
import os



#FILE LOCATIONS
model_name = 'BaselineNet_NoPad_Adam_batch64_64pix_3dense_lr001_E1000'
# data location
#data_folder = '/home/ubuntu/image_as_numpy/'
data_folder = '/home/ubuntu/image64/'
# output path
output_path = '/home/ubuntu/X-Ray-CNN/outputs'
monitor_path = '/home/ubuntu/X-Ray-CNN/monitor'

## epochs
epoch_number = 1000


# load the model
#model = ShallowNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
#model = BaselineNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
model = BaselineNet_NoPad.build(width = 64, height = 64, depth = 1, output = 1, dense_size = 500)
#model = AveragePoolingNet.build(width = 128, height = 128, depth = 1, output = 1)
#model = BaselineNet_LeakyRelu.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
#model = MicroVGGNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 2000)
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.001),
	metrics = ['binary_accuracy'])

model.summary()


# load the data
from keras.preprocessing.image import ImageDataGenerator

# load the train and validation data
X_train = np.load(data_folder + 'X_train.npy')
X_validate = np.load(data_folder + 'X_validate.npy')
y_train = np.load(data_folder + 'y_train.npy')
y_validate = np.load(data_folder + 'y_validate.npy')

X_test = np.load(data_folder + 'X_test.npy')
y_test = np.load(data_folder + 'y_test.npy')


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

train_generator = train_datagen.flow(X_train, y_train, batch_size = 64)
validation_generator = validate_datagen.flow(X_validate, y_validate, batch_size = 64)


## SAVING OUTPUTS AND LOGS
# create the image callback
figPath = os.path.sep.join([monitor_path, model_name + '.png'])
jsonPath = os.path.sep.join([monitor_path, model_name + '.json'])

# create checkpoint
#fname = os.path.sep.join([output_path, 'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
#checkpoint = ModelCheckpoint(fname, monitor = 'val_acc',mode = 'max', save_best_only = True, verbose = 1)
checkpoint = ModelCheckpoint(output_path + '/' +  model_name + '|_best_weights.hdf5', monitor = 'val_binary_accuracy',mode = 'max', save_best_only = True, verbose = 1)
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), checkpoint]

# TRAINING THE MODEL
history = model.fit_generator(train_generator,
                                  steps_per_epoch = len(X_train)/64, # 264 batches per epoch\n",
                                  epochs = epoch_number,
                                  validation_data = validation_generator,
                                  validation_steps = len(X_validate)/64,
                                  callbacks = callbacks)

model.save(output_path + '/' + model_name +  '|_final_result.hdf5')

## evaluate final model
train_datagen = ImageDataGenerator(rescale = 1./255)
validate_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen.fit(X_train)
validate_datagen.fit(X_validate)
test_datagen.fit(X_test)

train_generator = train_datagen.flow(X_train, y_train, batch_size = 64)
validate_generator = validate_datagen.flow(X_validate, y_validate, batch_size = 64)
test_generator = test_datagen.flow(X_test, y_test, batch_size = 64)

train_evaluate = model.evaluate_generator(train_generator, steps = len(X_train)/64)
validate_evaluate = model.evaluate_generator(validate_generator, steps = len(X_validate)/64)
test_evaluate = model.evaluate_generator(test_generator, steps = len(X_test)/64)

print('train_evaluate:', train_evaluate)
print('validate_evaluate:', validate_evaluate)
print('test_evaluate:', test_evaluate)


