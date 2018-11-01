# visualize model

from models import ShallowNet
from models import BaselineNet
from models import MicroVGGNet
from models import BaselineNet_NoPad
from models import BaselineNet_LeakyRelu
from models import AveragePoolingNet


model = AveragePoolingNet.build(width = 128, height = 128, depth = 1, output = 1)
model.summary()

# drawing the model
#from keras.utils import plot_model
#plot_model(model, to_file = 'ShallowNet.png', show_shapes = True)
