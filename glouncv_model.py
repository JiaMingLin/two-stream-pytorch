import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/ThrowDiscus.png'
im_fname = utils.download(url)

img = image.imread(im_fname)

transform_fn = transforms.Compose([
    video.VideoCenterCrop(size=256),
    video.VideoToTensor(),
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_list = transform_fn([img.asnumpy()])

net = get_model('resnet34_v1b_kinetics400', nclass=400, pretrained=True)

net.hybridize()

pred = net(nd.array(img_list[0]).expand_dims(axis=0))

net.export("resnet34_v1b_kinetics400_j")


