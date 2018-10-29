from keras import optimizers
from keras.callbacks import ModelCheckpoint
from models import BaselineNet
import argparse

import matplotlib
matplotlib.use('Agg')
from models.callbacks import TrainingMonitor
import os

# Create arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required = True,
	help = 'path to the best model weights file')
ap.add_argument('-m', '--monitor', required = True,
	help = 'path to the image output directory')
args = vars(ap.parse_args())


# load the model
model = BaselineNet.build(width = 128, height = 128, depth = 1, classes = 1, dense_size = 2000)
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4),
	metrics = ['accuracy'])

model.summary()


# load the data
from keras.preprocessing.image import ImageDataGenerator

train_dir = '/home/ubuntu/proper_train_test_split/train'
validation_dir = '/home/ubuntu/proper_train_test_split/validate'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

    
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode = 'grayscale',
                                                   target_size = (128, 128),
                                                   batch_size = 30,
                                                   class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        color_mode = 'grayscale',
                                                       target_size = (128, 128),
                                                       batch_size = 30,
                                                       class_mode = 'binary')


model_name = 'BaselineNet_DropoutDense_ImageAug'
# output path
output_path = args['output']
monitor_path = args['monitor']

# create the image callback
figPath = os.path.sep.join([monitor_path, model_name + '.png'])
jsonPath = os.path.sep.join([monitor_path, model_name + '.json'])

# create checkpoint
checkpoint = ModelCheckpoint(output_path + '/' +  model_name + '_best_weights.hdf5', monitor = 'val_acc',mode = 'max', save_best_only = True, verbose = 1)
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), checkpoint]

# train the model
history = model.fit_generator(train_generator,
                                  steps_per_epoch = 264, # 264 batches per epoch\n",
                                  epochs = 30,
                                  validation_data = validation_generator,
                                  validation_steps = 56,
                                  callbacks = callbacks)

model.save(output_path + '/' + model_name +  '_final_result.hdf5')
:w

