from __future__ import division
import random
import math
import numbers
import numpy as np
from mxnet.gluon import Block

from PIL import Image

__all__ = ['CustomizedTrainTransformation', 'CustomizedValTransformation']

class CustomizedTrainTransformation(Block):
    """Combination of transforms for training.
        (1) multiscale crop
        (2) scale
        (3) random horizontal flip
        (4) to tensor
        (5) normalize
    """
    def __init__(self, size, scale_ratios, mean, std, fix_crop=True,
                 more_fix_crop=True, max_distort=1, prob=0.5, max_intensity=255.0):
        super(CustomizedTrainTransformation, self).__init__()

        self.height = size[0]
        self.width = size[1]
        self.multiScaleCrop = VideoMultiScaleCrop(size=size,
                                                  scale_ratios=scale_ratios,
                                                  fix_crop=fix_crop,
                                                  more_fix_crop=more_fix_crop,
                                                  max_distort=max_distort)
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.prob = prob
        self.max_intensity = max_intensity
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))


    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.multiScaleCrop.fillCropSize(self.scale_ratios, self.max_distort, h, w)
        size_sel = random.choice(crop_size_pairs)
        crop_height = size_sel[0]
        crop_width = size_sel[1]

        is_flip = random.random() < self.prob
        if self.fix_crop:
            offsets = self.multiScaleCrop.fillFixOffset(self.more_fix_crop, h, w, crop_height, crop_width)
            off_sel = random.choice(offsets)
            h_off = off_sel[0]
            w_off = off_sel[1]
        else:
            h_off = random.randint(0, h - crop_height)
            w_off = random.randint(0, w - crop_width)

        new_clips = []
        for cur_img in clips:
            crop_img = Image.fromarray(cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :])
            # scale_img = self.cv2.resize(crop_img, (self.width, self.height))

            scale_img = np.array(crop_img.resize((self.width, self.height), Image.ANTIALIAS))
            if is_flip:
                flip_img = np.flip(scale_img, axis=1)
            else:
                flip_img = scale_img
            tensor_img = np.transpose(flip_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips


class CustomizedValTransformation(Block):
    """Combination of transforms for validation.
        (1) center crop
        (2) to tensor
        (3) normalize
    """

    def __init__(self, size, mean, std, max_intensity=255.0):
        super(CustomizedValTransformation, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))
        self.max_intensity = max_intensity

    def forward(self, clips):
        h, w, _ = clips[0].shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
            tensor_img = np.transpose(crop_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips

class VideoMultiScaleCrop(Block):
    """Corner cropping and multi-scale cropping.
        Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao
    Parameters:
    ----------
    size : int
        height and width required by network input, e.g., (224, 224)
    scale_ratios : list
        efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
    fix_crop : bool
        use corner cropping or not. Default: True
    more_fix_crop : bool
        use more corners or not. Default: True
    max_distort : float
        maximum aspect ratio distortion, used together with scale_ratios. Default: 1
    Inputs:
        - **data**: a list of frames with shape [H x W x C]
    Outputs:
        - **out**: a list of cropped frames with shape [size x size x C]
    """

    def __init__(self, size, scale_ratios, fix_crop=True,
                 more_fix_crop=True, max_distort=1):
        super(VideoMultiScaleCrop, self).__init__()
        # from ...utils.filesystem import try_import_cv2
        import cv2
        self.cv2 = cv2
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort

    def fillFixOffset(self, more_fix_crop, image_h, image_w, crop_h, crop_w):
        """Fixed cropping strategy. Only crop the 4 corners and the center.
        If more_fix_crop is turned on, more corners will be counted.
        Inputs:
            - **data**: height and width of input tensor
        Outputs:
            - **out**: a list of locations to crop the image
        """
        h_off = (image_h - crop_h) // 4
        w_off = (image_w - crop_w) // 4

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        if more_fix_crop:
            offsets.append((0, 2*w_off))        # top center
            offsets.append((4*h_off, 2*w_off))  # bottom center
            offsets.append((2*h_off, 0))        # left center
            offsets.append((2*h_off, 4*w_off))  # right center

            offsets.append((1*h_off, 1*w_off))  # upper left quarter
            offsets.append((1*h_off, 3*w_off))  # upper right quarter
            offsets.append((3*h_off, 1*w_off))  # lower left quarter
            offsets.append((3*h_off, 3*w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, scale_ratios, max_distort, image_h, image_w):
        """Fixed cropping strategy. Select crop size from
        pre-defined list (computed by scale_ratios).
        Inputs:
            - **data**: height and width of input tensor
        Outputs:
            - **out**: a list of crop sizes to crop the image
        """
        crop_sizes = []
        base_size = np.min((image_h, image_w))
        for h_index, scale_rate_h in enumerate(scale_ratios):
            crop_h = int(base_size * scale_rate_h)
            for w_index, scale_rate_w in enumerate(scale_ratios):
                crop_w = int(base_size * scale_rate_w)
                if (np.absolute(h_index - w_index) <= max_distort):
                    # To control the aspect ratio distortion
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.fillCropSize(self.scale_ratios, self.max_distort, h, w)
        size_sel = random.choice(crop_size_pairs)
        crop_height = size_sel[0]
        crop_width = size_sel[1]

        if self.fix_crop:
            offsets = self.fillFixOffset(self.more_fix_crop, h, w, crop_height, crop_width)
            off_sel = random.choice(offsets)
            h_off = off_sel[0]
            w_off = off_sel[1]
        else:
            h_off = random.randint(0, h - crop_height)
            w_off = random.randint(0, w - crop_width)

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            new_clips.append(self.cv2.resize(crop_img, (self.width, self.height)))
        return new_clips