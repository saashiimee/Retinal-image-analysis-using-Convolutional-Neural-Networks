import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from config.config import prepare_config
from src.data import DataLoader
from src.model_unet import UNET
from src.train import ModelTrain
from src.utils.metrics_utils import dice_coeff, IOU


def main_train():
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

    data_loader = DataLoader(config=config)
    data_loader.prepare_dataset()
    print('[INFO] Preparing Data...')

    train_imgs, train_groundtruth = data_loader.get_train_data()
    validate_imgs, validate_groundtruth = data_loader.get_validate_data()

    print('[INFO] Building Model...')
    with strategy.scope():
        roc_curve = AUC(curve='ROC', name='ROC')
        pr_curve = AUC(curve='PR', name='PR')
        model = UNET(config=config)
        model.build_model(
            input_shape=(config.desired_size, config.desired_size, config.num_channel),
            num_classes=config.num_class,
            dropout=config.dropout,
            output_activation="sigmoid"
        )
        model.compile_model(
            optimizer=Adam,
            learning_rate=config.learning_rate,
            loss='binary_crossentropy',
            metrics=['accuracy', dice_coeff, IOU, roc_curve, pr_curve]
        )
        model.save()

    print('[INFO] Training...')
    train_model = ModelTrain(
        model=model.model,
        data=[train_imgs, train_groundtruth, validate_imgs, validate_groundtruth],
        config=config)
    train_model.train()
    print('[INFO] Finishing...')


if __name__ == '__main__':
    main_train()
