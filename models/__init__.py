from .rgb_vgg16 import *
from .flow_vgg16 import *
from .rgb_resnet import *
from .flow_resnet import *

try:
    from .flow_resnet_mx import *
except:
  	print("No mxnet installed")
