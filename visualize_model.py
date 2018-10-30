# visualize model

from models import ShallowNet
from models import BaselineNet
from models import MicroVGGNet


model = BaselineNet.build(width = 128, height = 128, depth = 1, output = 1, dense_size = 5000)
model.summary()

# drawing the model
#from keras.utils import plot_model
#plot_model(model, to_file = 'ShallowNet.png', show_shapes = True)
