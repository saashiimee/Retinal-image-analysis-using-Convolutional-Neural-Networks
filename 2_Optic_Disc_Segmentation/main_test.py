import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

from statistics import *
from src.test import ModelPredict
from src.utils.metrics_utils import IOU, dice_coeff
from config.config import prepare_config


def main_test():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1"],
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )

    print('[INFO] Reading Configs...')
    config = None

    try:
        config = prepare_config('config/config.json')
    except Exception as e:
        print('[Error] Config Error, %s' % e)
        exit(0)

    print('[INFO] Predicting...')
    model_predict = ModelPredict(config)
    with CustomObjectScope({'dice_score': dice_coeff}, {'IOU': IOU}):
        model_predict.load_model()
    model_predict.predict()

    print('[INFO] Metric results...')
    gt_list = fileList(config.test_groundtruth_path, '*')
    prob_list = fileList(config.test_result_path, '*')
    model_name = [config.model_name]
    drawCurve(gt_list, [prob_list], model_name, config.dataset_name, config, config.checkpoint)

    print('[INFO] Finished...')


if __name__ == '__main__':
    main_test()
