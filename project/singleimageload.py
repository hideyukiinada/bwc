#!/usr/bin/env python
'''Load a single image file.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"
'''

import os
import logging

from pathlib import Path
from PIL import Image
import numpy as np
import random

from project.config import load_config
from project.maddiecnncommon import DogClassIndex
from project.maddiecnncommon import DogClassMarker

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging
config = load_config()


def load_single_image_by_path_with_size_validation(path, height_expected, width_expected, channels_expected):
    """Load a single Pillow image from file

    Parameters
    ----------
    path: pathlib.Path
        Path to the image file on the file system
    height_expected: int
        Image height expected
    width_expected: int
        Image width expected
    channels_expected: int
        Number of channels in the image expected

    Returns
    -------
    data: numpy array
        Image data
    """

    image_pil = Image.open(path)
    image_size = image_pil.size

    if image_size[0] != width_expected:
        log.warning("Unexpected width %d for image file %s. Expecting: %d" % (image_size[0], path, width_expected))
        return None

    if image_size[1] != height_expected:
        log.warning(
            "Unexpected height %d for image file %s. Expecting: %d" % (image_size[1], path, height_expected))
        return None

    data = np.array(image_pil)
    if len(data.shape) < 3:
        channels = 1
    else:
        channels = data.shape[2]
    if channels != channels_expected:
        log.warning(
            "Unexpected number of channels %d for image file %s. Expecting: %d" % (channels, path, channels_expected))
        return None

    data = data.reshape(height_expected, width_expected, channels_expected)
    return data
