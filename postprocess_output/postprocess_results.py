import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
## Post processing

## load the models
model_path = '/home/ubuntu/X-Ray-CNN/outputs/BaselineNet_NoPad_Adam_batch64_E150|_best_weights.hdf5'

model = load_model(model_path)

## load train, validate and test data
folder_path = '/home/ubuntu/image_as_numpy/'

X_train = np.load(folder_path + 'X_train.npy')
X_validate = np.load(folder_path + 'X_validate.npy')
X_test = np.load(folder_path + 'X_test.npy')

y_train = np.load(folder_path + 'y_train.npy')
y_validate = np.load(folder_path + 'y_validate.npy')
y_test = np.load(folder_path + 'y_test.npy')

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


## perform predict on all 3 X 
y_train_pred = model.predict_generator(train_generator, steps = len(X_train)/64)
y_validate_pred = model.predict_generator(validate_generator, steps = len(X_validate)/64)
y_test_pred = model.predict_generator(test_generator, steps = len(X_test)/64)

## save the evaluation and y_preds and predict proba as numpy arrays
output_path = '/home/ubuntu/X-Ray-CNN/postprocess_output/BaselineNet_NoPad_Adam_batch64_E150/'
np.save(output_path + 'train_evaluate.npy', train_evaluate)
np.save(output_path + 'validate_evaluate.npy', validate_evaluate)
np.save(output_path + 'test_evaluate.npy', test_evaluate)

np.save(output_path + 'y_train_pred.npy', y_train_pred)
np.save(output_path + 'y_validate_pred.npy', y_validate_pred)
np.save(output_path + 'y_test_pred.npy', y_test_pred)
