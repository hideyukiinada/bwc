#!/usr/bin/env python
'''Load image files.

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
from project.singleimageload import load_single_image_by_path_with_size_validation

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging
config = load_config()


def generate_target_image_path_from_source_image_path(f, target_dir):
    parent = f.parent
    stem = f.stem
    suffix = f.suffix
    truncate_len = len("_gray")
    target = target_dir / Path(stem[0:-truncate_len] + suffix)
    return target


def load_image_data(test_set_only=False):
    """Load image files from the dataset directory

    Parameters
    ----------
    test_set_only: bool
        Load test set from a separate directory

    Returns
    -------
    source_image_list: list
        List of source images in grayscale
    target_image_list: list
        List of target images in color to train the model
    file_name_list: list
        Image file names ordered to match the sequence of image_list
    """

    if test_set_only:
        source_dataset_path = Path(config["source_test_set_directory"])
        target_dataset_path = Path(config["predicted_image_directory"])
    else:
        source_dataset_path = Path(config["source_dataset_directory"])
        target_dataset_path = Path(config["target_dataset_directory"])

        log.debug("Loading source data set from %s" % (source_dataset_path))
        log.debug("Loading target data set from %s" % (target_dataset_path))

    source_image_list = list()
    target_image_list = list()

    files = list(source_dataset_path.glob("*.jpg"))
    num_files = len(files)
    log.debug("Number of image files found: %d" % (num_files))

    source_file_name_list = list()
    target_file_name_list = list()

    dataset_size = 0
    for i, f_source in enumerate(files):

        src_data = load_single_image_by_path_with_size_validation(f_source,
                                                                  height_expected=config["image_height"],
                                                                  width_expected=config["image_width"],
                                                                  channels_expected=config["number_of_channels_in_source_image"]
                                                                  )

        if src_data is None:
            log.warning(
                "Skipping invalid image file name or data: %s" % (f_source))
            continue

        f_target = generate_target_image_path_from_source_image_path(f_source, target_dataset_path)

        if test_set_only is False:
            if f_target.exists() is False:
                log.warning("Not found target image: %s" % (f_target))
                continue

            target_data = load_single_image_by_path_with_size_validation(f_target,
                                                                         height_expected=config["image_height"],
                                                                         width_expected=config["image_width"],
                                                                         channels_expected=config["number_of_channels_in_target_image"]
                                                                         )
            if target_data is None:
                log.warning(
                    "Skipping invalid image file name or data: %s" % (f_target))
                continue

        source_image_list.append(src_data)
        if test_set_only is False:
            target_image_list.append(target_data)

        source_file_name_list.append(f_source.name)
        target_file_name_list.append(f_target.name)

        dataset_size = dataset_size + 1

        if i + 1 >= config["max_number_of_files_to_load"] and config["max_number_of_files_to_load"] != -1:
            break

    log.debug("Number of image files loaded: %d" % (dataset_size))

    return source_image_list, target_image_list, source_file_name_list, target_file_name_list


def load_training_set_and_test_set(test_data_ratio=0.2):
    """Load image files from the dataset directory

    Parameters
    ----------
    test_data_ratio: float
        Ratio of test data size in the total dataset size

    Returns
    -------
    x: numpy array
        Training set data
    y: numpy array
        Training set ground truth values
    x_test: numpy array
        Test set data
    y_test: numpy array
        Test set ground truth values
    file_name_training_list: list
        Training set image file names ordered to match the sequence of training set
    file_name_test_list: list
        Test set image file names ordered to match the sequence of training set
    """

    source_image_list, target_image_list, source_file_name_list, target_file_name_list = load_image_data()

    dataset_size = len(source_image_list)

    file_name_training_list = list()
    file_name_test_list = list()

    training_dataset_size = int(dataset_size * (1 - test_data_ratio))
    test_dataset_size = dataset_size - training_dataset_size

    x = np.zeros((training_dataset_size, config["image_height"], config["image_width"],
                  config["number_of_channels_in_source_image"]), dtype='float32')
    y = np.zeros((training_dataset_size, config["image_height"], config["image_width"],
                  config["number_of_channels_in_target_image"]), dtype='float32')

    x_test = np.zeros((test_dataset_size, config["image_height"], config["image_width"],
                       config["number_of_channels_in_source_image"]), dtype='float32')
    y_test = np.zeros((test_dataset_size, config["image_height"], config["image_width"],
                       config["number_of_channels_in_target_image"]), dtype='float32')

    # Shuffle data
    random_index_array = list(range(dataset_size))
    random.shuffle(random_index_array)  # shuffle data inplace

    for i in range(training_dataset_size):
        single_image = source_image_list[random_index_array[i]]
        x[i] = single_image / 255
        single_image = target_image_list[random_index_array[i]]
        y[i] = single_image / 255
        file_name_training_list.append(source_file_name_list[random_index_array[i]])

    for i in range(test_dataset_size):
        single_image = source_image_list[random_index_array[i + training_dataset_size]]
        x_test[i] = single_image / 255

        single_image = target_image_list[random_index_array[i + training_dataset_size]]
        y_test[i] = single_image / 255
        file_name_test_list.append(source_file_name_list[random_index_array[i + training_dataset_size]])

    log.debug("Data loaded")
    return x, y, x_test, y_test, file_name_training_list, file_name_test_list


def load_test_set():
    """Load image files from the test set directory


    Returns
    -------
    x_test: numpy array
        Test set data
    y_test: numpy array
        Test set ground truth values
    source_file_name_list: list
        Test set image file names containing the source image
    target_file_name_list: list
        Test set image file names to be used to save the predicted image file
    """

    source_image_list, target_image_list, source_file_name_list, target_file_name_list = load_image_data(test_set_only=True)

    dataset_size = len(source_image_list)

    file_name_test_list = list()

    x_test = np.zeros((dataset_size, config["image_height"], config["image_width"],
                       config["number_of_channels_in_source_image"]), dtype='float32')

    for i in range(dataset_size):
        single_image = source_image_list[i]
        x_test[i] = single_image / 255

    log.debug("Test set loaded")
    return x_test, source_file_name_list, target_file_name_list
