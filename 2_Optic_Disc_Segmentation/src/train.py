import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from src.base.train_base import TrainBase
from src.utils.img_utils import pre_process
from src.utils.plot_utils import plot_metric


class ModelTrain(TrainBase):
    def __init__(self, model, data, config):
        super(ModelTrain, self).__init__(model, data, config)
        self.model = model
        self.data = data
        self.config = config
        self.callbacks = []
        self.init_callbacks()

        self.train_img = pre_process(data[0], config.desired_size)
        self.train_gt = data[1]
        self.val_img = pre_process(data[2], config.desired_size)
        self.val_gt = data[3]

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath='./models/' + self.config.dataset_name + '_best_weights.h5',
                verbose=1,
                monitor='val_loss',
                mode='auto',
                save_best_only=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.checkpoint + "logs",
                write_images=True,
                write_graph=True,
            )
        )

        # self.callbacks.append(
        #     EarlyStopping(
        #         monitor='val_loss',
        #         patience=5,
        #         verbose=0,
        #         mode='auto'
        #     )
        # )

        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
            )
        )

    def train(self):
        train_data = tf.data.Dataset.from_tensor_slices((self.train_img, self.train_gt))
        val_data = tf.data.Dataset.from_tensor_slices((self.val_img, self.val_gt))

        train_data = train_data.batch(self.config.batch_size)
        val_data = val_data.batch(self.config.batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        history = self.model.fit(train_data,
                                 epochs=self.config.epochs,
                                 batch_size=self.config.batch_size,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 validation_data=val_data,
                                 )
        self.model.save_weights(self.config.checkpoint + self.config.dataset_name + '_last_weights.h5', overwrite=True)
        plot_metric(history, 'loss', self.config.checkpoint)
        plot_metric(history, 'accuracy', self.config.checkpoint)
        plot_metric(history, 'dice_coeff', self.config.checkpoint)
        plot_metric(history, 'IOU', self.config.checkpoint)
        plot_metric(history, 'ROC', self.config.checkpoint)
        plot_metric(history, 'PR', self.config.checkpoint)

