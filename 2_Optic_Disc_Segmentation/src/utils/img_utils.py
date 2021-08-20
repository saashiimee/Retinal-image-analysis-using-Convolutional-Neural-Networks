
import numpy as np
import cv2


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image, table)


def clahe_equalized(image):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def pre_process(images, desired_size):
    preprocessed_images = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = clahe_equalized(image)
        image = adjust_gamma(image, 1.1)
        image = image.astype(np.float32)
        image = image / 255.
        preprocessed_images.append(image)

    preprocessed_images = np.array(preprocessed_images)
    preprocessed_images = np.reshape(preprocessed_images, (len(preprocessed_images), desired_size, desired_size, 1))

    return preprocessed_images
