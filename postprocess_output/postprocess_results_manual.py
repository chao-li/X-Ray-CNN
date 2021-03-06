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

# proba
y_train_proba = model.predict_proba(X_train, batch_size = 64)
y_validate_proba = model.predict_proba(X_validate, batch_size = 64)
y_test_proba = model.predict_proba(X_test, batch_size = 64)


## save the evaluation and y_preds and predict proba as numpy arrays
output_path = '/home/ubuntu/X-Ray-CNN/postprocess_output/BaselineNet_NoPad_Adam_batch64_E150/'
np.save(output_path + 'train_evaluate.npy', train_evaluate)
np.save(output_path + 'validate_evaluate.npy', validate_evaluate)
np.save(output_path + 'test_evaluate.npy', test_evaluate)

np.save(output_path + 'y_train_pred.npy', y_train_pred)
np.save(output_path + 'y_validate_pred.npy', y_validate_pred)
np.save(output_path + 'y_test_pred.npy', y_test_pred)

np.save(output_path + 'y_train_proba.npy', y_train_proba)
np.save(output_path + 'y_validate_proba.npy', y_validate_proba)
np.save(output_path + 'y_test_proba.npy', y_test_proba)
