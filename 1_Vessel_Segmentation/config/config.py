import os
import json

from bunch import Bunch

from src.utils.data_utils import mkdir_if_not_exist


def get_config_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config, config_dict


def prepare_config(json_file):
    config, _ = get_config_json(json_file)

    config.checkpoint = os.path.join("./data", config.dataset_name, "checkpoint/")
    config.hdf5_path = os.path.join("./data", config.dataset_name, "hdf5/")
    config.train_img_path = os.path.join("./data", config.dataset_name, "train/images/")
    config.train_groundtruth_path = os.path.join("./data", config.dataset_name, "train/groundtruth/")
    config.validate_img_path = os.path.join("./data", config.dataset_name, "validate/images/")
    config.validate_groundtruth_path = os.path.join("./data", config.dataset_name, "validate/groundtruth/")
    config.test_img_path = os.path.join("./data", config.dataset_name, "test/images/")
    config.test_groundtruth_path = os.path.join("./data", config.dataset_name, "test/groundtruth/")
    config.test_result_path = os.path.join("./data", config.dataset_name, "test/result/")

    mkdir_if_not_exist(config.checkpoint)
    mkdir_if_not_exist(config.hdf5_path)
    mkdir_if_not_exist(config.train_img_path)
    mkdir_if_not_exist(config.train_groundtruth_path)
    mkdir_if_not_exist(config.validate_img_path)
    mkdir_if_not_exist(config.validate_groundtruth_path)
    mkdir_if_not_exist(config.test_img_path)
    mkdir_if_not_exist(config.test_groundtruth_path)
    mkdir_if_not_exist(config.test_result_path)

    return config
