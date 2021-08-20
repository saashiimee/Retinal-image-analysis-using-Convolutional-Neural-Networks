import os

import tensorflow as tf
# from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    UpSampling2D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
)

from src.base.model_base import ModelBase


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type="spatial",
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
):
    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}")

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


class UNET(ModelBase):
    def __init__(self, config=None):
        super(UNET, self).__init__(config)

    def build_model(
            self,
            input_shape,
            num_classes=1,
            activation="relu",
            use_batch_norm=True,
            upsample_mode="deconv",
            dropout=0.1,
            dropout_change_per_layer=0.0,
            dropout_type="spatial",
            use_dropout_on_upsampling=False,
            use_attention=True,
            filters=64,
            num_layers=4,
            output_activation="sigmoid",
    ):

        if upsample_mode == "deconv":
            upsample = upsample_conv
        else:
            upsample = upsample_simple

        inputs = Input(input_shape)
        x = inputs

        down_layers = []
        for l in range(num_layers):
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters = filters * 2

        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0

        for conv in reversed(down_layers):
            filters //= 2
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            if use_attention:
                x = attention_concat(conv_below=x, skip_connection=conv)
            else:
                x = concatenate([x, conv])
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )

        outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        self.model = model

    def compile_model(
            self,
            optimizer=Adam,
            learning_rate=1e-3,
            loss='binary_crossentropy',
            metrics=None):

        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
