# visualize model

from models import ShallowNet
from models import ShallowNet128
from models import BaselineNet
from models import BaselineNet_Reg
from models import MicroVGGNet
from models import BaselineNet_NoPad
from models import BaselineNet_LeakyRelu
from models import AveragePoolingNet


model = BaselineNet_NoPad.build(width = 64, height = 64, depth = 1, output = 1)
model.summary()

# drawing the model
#from keras.utils import plot_model
#plot_model(model, to_file = 'ShallowNet.png', show_shapes = True)
