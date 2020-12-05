import mxnet as mx
from mxnet import init
from mxnet import ndarray as nd
from mxnet.util import is_np_array
from mxnet.gluon import nn
from mxnet.gluon.utils import _brief_print_list
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.resnetv1b import resnet18_v1b, resnet34_v1b
import numpy as np
from collections import OrderedDict, defaultdict
__all__ = ['resnet18_v1b_kinetics400',
           'resnet34_v1b_kinetics400']

class ActionRecResNetV1bCustom(HybridBlock):
    r"""ResNet models for video action recognition
    Deep Residual Learning for Image Recognition, CVPR 2016
    https://arxiv.org/abs/1512.03385
    Parameters
    ----------
    depth : int, default is 50.
        Depth of ResNet, from {18, 34, 50, 101, 152}.
    nclass : int
        Number of classes in the training dataset.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    init_std : float, default is 0.001.
        Standard deviation value when initialize the dense layers.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    Input: a single video frame or N images from N segments when num_segments > 1
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,
                 dropout_ratio=0.5, init_std=0.01,
                 num_segments=1, num_crop=1,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1bCustom, self).__init__()

        modality = kwargs['modality']
        in_channels = kwargs['in_channels']

        if depth == 18:
            pretrained_model = resnet18_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 34:
            pretrained_model = resnet34_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        else:
            print('No such ResNet configuration for depth=%d' % (depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_dim = 512 * self.expansion
        self.num_segments = num_segments
        self.num_crop = num_crop

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False, in_channels = in_channels,
                                       weight_initializer=init.Normal(sigma=self.init_std))
            self.conv1.initialize()
            self.bn1 = pretrained_model.bn1
            self.relu = pretrained_model.relu
            self.maxpool = pretrained_model.maxpool
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
            self.avgpool = pretrained_model.avgpool
            self.flat = pretrained_model.flat
            self.drop = nn.Dropout(rate=self.dropout_ratio)
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                                   weight_initializer=init.Normal(sigma=self.init_std))
            self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

    def load_dict(self, param_dict, ctx=None, allow_missing=True,
                  ignore_extra=True, cast_dtype=False, dtype_source="current"):
        """Load parameters from dict
        Parameters
        ----------
        param_dict : dict
            Dictionary containing model parameters
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represented in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this dict.
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        """
        if isinstance(param_dict.get('filename'), str):
            # pass from load_parameters
            filename = param_dict['filename']
            param_dict = param_dict['params']
        else:
            filename = None
        params = self.collect_params()
        error_str = "file: %s" % (filename) if filename else "param_dict"
        loaded = {k[4:] if k.startswith('arg:') or k.startswith('aux:') else k: v \
                  for k, v in param_dict.items()}

        if not allow_missing:
            params_inv = defaultdict(list)
            for k, v in params.items():
                params_inv[v].append(k)

            for name, param in params.items():
                assert any(p in loaded for p in params_inv[param]), \
                    "Parameter '%s' is missing in '%s', which contains parameters: %s. " \
                    "Set allow_missing=True to ignore missing parameters."%(
                        name, error_str, _brief_print_list(loaded.keys()))

        if ctx is None:
            ctx = _context.current_context()
        for name in loaded:
            if not ignore_extra and name not in params:
                raise ValueError(
                    "Parameter '%s' loaded from '%s' is not present in Dict, " \
                    "which contains parameters %s. Set ignore_extra=True to ignore. "%(
                        name, error_str, _brief_print_list(params.keys())))
            if name in params:
                param = loaded[name]
                if isinstance(param, np.ndarray):
                    param = np.array(param) if is_np_array() else nd.array(param)
                params[name]._load_init(param, ctx, cast_dtype=cast_dtype, dtype_source=dtype_source)


def change_key_names(old_params, in_channels):
    new_params = OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count >= len(allKeyList)-2:
            # exclude fc layers
            continue
        else:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                # print(type(rgb_weight))
                rgb_weight_mean = rgb_weight.mean(axis=1)
                rgb_weight_mean = np.transpose(rgb_weight_mean.reshape((-1,)+rgb_weight_mean.shape), (1,0,2,3))
                flow_weight = rgb_weight_mean.repeat(repeats = 20, axis = 1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
    
    return new_params


"""
net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    use_tsn=opt.use_tsn, num_segments=opt.num_segments, partial_bn=opt.partial_bn,
                    bn_frozen=opt.freeze_bn)
"""

def resnet18_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    r"""ResNet18 model trained on Kinetics400 dataset.
    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    """

    modality = kwargs['modality']

    in_channels = 3 if modality == 'rgb' else 20
    model = ActionRecResNetV1bCustom(depth=18,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01, 
                               modality = modality,
                               in_channels = in_channels)

    if pretrained:

        from gluoncv.model_zoo.model_store import get_model_file
        params_dict = mx.nd.load(get_model_file('resnet18_v1b_kinetics400', tag=pretrained, root=root))
        if modality == 'tvl1_flow':
            params_dict = change_key_names(params_dict, in_channels=20)

        model.load_dict(params_dict, ctx=ctx)
        from gluoncv.data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             use_tsn=False, partial_bn=False,
                             num_segments=1, num_crop=1, root='~/.mxnet/models',
                             ctx=mx.cpu(), **kwargs):
    r"""ResNet34 model trained on Kinetics400 dataset.
    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    """
    modality = kwargs['modality']
    model = ActionRecResNetV1bCustom(depth=34,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        model.load_parameters(get_model_file('resnet34_v1b_kinetics400',tag=pretrained, root=root))
        from gluoncv.data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model