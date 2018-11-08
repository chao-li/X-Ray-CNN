from keras import optimizers
from keras.callbacks import ModelCheckpoint
from models import BaselineNet
from models import BaselineNet_Reg
from models import ShallowNet
from models import MicroVGGNet
from models import BaselineNet_NoPad
from models import BaselineNet_LeakyRelu
from models import AveragePoolingNet
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
from models.callbacks import TrainingMonitor
import os

# Baseline Reg


#FILE LOCATIONS
model_name = 'BaselineNet_NoPad_Reg_150Epoch'
# data location
#data_folder = '/home/ubuntu/image_as_numpy/'
data_folder = '/home/ubuntu/image64/'
# output path
output_path = '/home/ubuntu/X-Ray-CNN/outputs'
monitor_path = '/home/ubuntu/X-Ray-CNN/monitor'

## epochs
epoch_number = 150


# load the model
model = BaselineNet_NoPad.build(width = 64, height = 64, depth = 1, output = 1)

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.0001),
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
model_path = output_path + '/' +  model_name + '.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor = 'val_binary_accuracy',mode = 'max', save_best_only = True, verbose = 1)
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), checkpoint]

# TRAINING THE MODEL
history = model.fit_generator(train_generator,
                                  steps_per_epoch = len(X_train)/64, # 264 batches per epoch\n",
                                  epochs = epoch_number,
                                  validation_data = validation_generator,
                                  validation_steps = len(X_validate)/64,
                                  callbacks = callbacks)



## evaluate final model
model = load_model(model_path)

## normalize X
X_train = X_train * (1./255)
X_validate = X_validate * (1./255)
X_test = X_test * (1./255)

## evaluate
train_evaluate = model.evaluate(X_train, y_train, batch_size = 64)
validate_evaluate = model.evaluate(X_validate, y_validate, batch_size = 64)
test_evaluate = model.evaluate(X_test, y_test, batch_size = 64)

print('train evaluate:', train_evaluate)
print('validate evaluate:', validate_evaluate)
print('test evaluate:', test_evaluate)

## predict
y_train_pred = model.predict(X_train, batch_size = 64)
y_validate_pred = model.predict(X_validate, batch_size = 64)
y_test_pred = model.predict(X_test, batch_size = 64)

## save the evaluation and y_preds and predict proba as numpy arrays
output_path = '/home/ubuntu/X-Ray-CNN/postprocess_output/all_models/'
np.save(output_path + model_name + '__train_evaluate.npy', train_evaluate)
np.save(output_path + model_name + '__validate_evaluate.npy', validate_evaluate)
np.save(output_path + model_name + '__test_evaluate.npy', test_evaluate)

np.save(output_path + model_name + '__y_train_pred.npy', y_train_pred)
np.save(output_path + model_name + '__y_validate_pred.npy', y_validate_pred)
np.save(output_path + model_name + '__y_test_pred.npy', y_test_pred)


