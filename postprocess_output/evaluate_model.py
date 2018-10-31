import numpy as np
from keras.models import load_model
## Post processing

## load the models
model_path = '/home/ubuntu/X-Ray-CNN/outputs/BaselineNet_ADAM_Epoch150_eyeballing|_best_weights.hdf5'

model = load_model(model_path)

## load train, validate and test data
folder_path = '/home/ubuntu/image_as_numpy/'

X_train = np.load(folder_path + 'X_train.npy')
X_validate = np.load(folder_path + 'X_validate.npy')
X_test = np.load(folder_path + 'X_test.npy')

y_train = np.load(folder_path + 'y_train.npy')
y_validate = np.load(folder_path + 'y_validate.npy')
y_test = np.load(folder_path + 'y_test.npy')

## perform evaluate on all 3 sets
train_evaluate = model.evaluate(X_train, y_train, batch_size = 32)
#validate_evaluate = model.evaluate(X_validate, y_validate, batch_size = 32)
#test_evaluate = model.evaluate(X_test, y_test, batch_size = 32)

#print('train', train_evaluate)
#print('validate', validate_evalutate)
print('test', test_evaluate)

