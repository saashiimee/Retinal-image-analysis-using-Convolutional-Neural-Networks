import glob
import numpy as np

from PIL import Image

from src.base.data_base import DataLoaderBase
from src.utils.data_utils import load_hdf5, write_hdf5


class DataLoader(DataLoaderBase):
    def __init__(self, config=None):
        super(DataLoader, self).__init__(config)
        self.dataset_name = config.dataset_name
        self.hdf5_path = config.hdf5_path
        self.desired_size = config.desired_size

        self.train_img_path = config.train_img_path
        self.train_groundtruth_path = config.train_groundtruth_path
        self.validate_img_path = config.validate_img_path
        self.validate_groundtruth_path = config.validate_groundtruth_path

    def read_images(self, paths):
        images = []
        for path in paths:
            img = Image.open(path)
            img = img.resize((self.desired_size, self.desired_size))
            img = np.array(img)
            images.append(img)
        return images

    def read_masks(self, paths):
        masks = []
        for path in paths:
            mask = Image.open(path).convert("L")
            mask = mask.resize((self.desired_size, self.desired_size))
            mask = np.array(mask)
            mask = mask / 255.
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask = mask.astype(np.float32)
            masks.append(mask)
        return masks

    def access_dataset(self, image_path, ground_truth_path):
        images_list = sorted(glob.glob(image_path + "*"))
        ground_truth_list = sorted(glob.glob(ground_truth_path + "*"))

        images = self.read_images(images_list)
        ground_truths = self.read_masks(ground_truth_list)

        images = np.array(images)
        ground_truths = np.array(ground_truths)

        images = np.reshape(images, (len(images), self.desired_size, self.desired_size, 3))
        ground_truths = np.reshape(ground_truths, (len(ground_truths), self.desired_size, self.desired_size, 1))

        print("[INFO] Reading Data...")

        return images, ground_truths

    def prepare_dataset(self):
        train_imgs, groundtruth = self.access_dataset(self.train_img_path, self.train_groundtruth_path)
        write_hdf5(train_imgs, self.hdf5_path + "/train_img.hdf5")
        write_hdf5(groundtruth, self.hdf5_path + "/train_groundtruth.hdf5")
        print("[INFO] Saving Training Data...")

        validate_imgs, groundtruth = self.access_dataset(self.validate_img_path, self.validate_groundtruth_path)
        write_hdf5(validate_imgs, self.hdf5_path + "/validate_img.hdf5")
        write_hdf5(groundtruth, self.hdf5_path + "/validate_groundtruth.hdf5")
        print("[INFO] Saving Validation Data...")
        print("[INFO] Data Saved")

    def get_train_data(self):
        train_imgs = load_hdf5(self.hdf5_path + "/train_img.hdf5")
        train_groundtruth = load_hdf5(self.hdf5_path + "/train_groundtruth.hdf5")
        return train_imgs, train_groundtruth

    def get_validate_data(self):
        validate_imgs = load_hdf5(self.hdf5_path + "/validate_img.hdf5")
        validate_groundtruth = load_hdf5(self.hdf5_path + "/validate_groundtruth.hdf5")
        return validate_imgs, validate_groundtruth
