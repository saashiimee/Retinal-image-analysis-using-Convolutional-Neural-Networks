import os
import glob
import math

import cv2
import numpy as np

from PIL import Image
from sklearn.metrics import  recall_score, roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

from statistics import *
from src.data import DataLoader
from src.base.test_base import TestBase
from src.utils.data_utils import mkdir_if_not_exist
from src.utils.img_utils import clahe_equalized, adjust_gamma, pre_process


def test_generator(test_path, target_size):
    files = sorted(os.listdir(test_path))
    num_image = len(files)
    for i in range(num_image):
        image = Image.open(os.path.join(test_path, files[i]))
        image = image.resize(target_size)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = clahe_equalized(image)
        image = adjust_gamma(image, 1.1)
        image = image.astype(np.float32)
        image = image / 255.
        image = np.reshape(image, (1,) + image.shape)
        yield image


def save_results(save_path, npyfile):
    mkdir_if_not_exist(save_path)
    for i, item in enumerate(npyfile):
        image = item * 255
        image = image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'result_{i}_predict.png'), image)


class ModelPredict(TestBase):
    def __init__(self, config):
        super(ModelPredict, self).__init__(config)
        self.config = config
        self.model = model_from_json(
            open('./models/' + self.config.model_name + '_' + self.config.dataset_name + '_architecture.json').read())

    def load_model(self):
        self.model.load_weights(
            './models/' + self.config.model_name + '_' + self.config.dataset_name + '_best_weights.h5')

    def analyze_name(self, path):
        return (path.split('\\')[-1]).split(".")[0]

    def predict(self):
        gen = test_generator(
            self.config.test_img_path,
            (self.config.desired_size, self.config.desired_size)
        )
        segment_results = self.model.predict(gen, batch_size=1, verbose=1)
        save_results(self.config.test_result_path, segment_results)
        evaluate_result(self.config.test_result_path, segment_results)

