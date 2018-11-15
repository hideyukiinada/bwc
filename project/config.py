#!/usr/bin/env python
'''Load config in JSON

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"
'''

import simplejson as json


def load_config():
    """Load config in JSON.

    Returns
    -------
    config: dictionary
        Configuration with key value pairs

    Note
    ----
    This function removes comments in JSON data for the caller to be able to access the config value without specifying
    the "value" key.
    """

    with open("config.json") as f:
        config_raw = json.load(f)

    # Remove comments
    config = dict()
    for k, v in config_raw.items():
        config[k] = config_raw[k]["value"]

    return config
