from typing import Optional

import tensorflow as tf
from keras.backend import ctc_batch_cost
from keras.layers import AdditiveAttention
from keras.layers import Concatenate, Dense, Input, Layer
from keras.models import Model

from CRNN.model.backbone import get_cnn_backbone
from CRNN.model.backbone import get_rnn_backbone


def CRNN(image_width: int,
         image_height: int,
         image_channels: int,
         cnn_backbone: Optional[str],
         rnn_backbone: Optional[str],
         characters_size: int,
         trainable: bool = True,
         use_attention: bool = False) -> Model:
    """Instantiates the CRNN model.

    Args:
        image_width: int,
            the width of image.
        image_height: int,
            the height of image.
        image_channels: {1, 3},
            the channels of image.
        cnn_backbone: {'resnet', 'efficientnet', 'plain_cnn'},
            the name of convolution part of CRNN model.
        rnn_backbone: {'bi_gru', 'plain_rnn'},
            the name of recurrent part of CRNN model.
        characters_size: int,
            the size of the characters in the dataset.
        trainable: bool, default=True,
            the model can be trained.
        use_attention: bool, default=False,
            the model has attention layers.

    Return:
        The CRNN model.
    """
    input_image = Input(shape=(image_width, image_height, image_channels), name='Input-Image', dtype='float32')
    input_label = Input(shape=(None, ), name='Input-Label', dtype='float32')

    # The CNN backbone.
    cnn_backbone = get_cnn_backbone(cnn_backbone)
    cnn_layer = cnn_backbone(input_image)

    # The RNN backbone.
    rnn_backbone = get_rnn_backbone(rnn_backbone)
    rnn_layer = rnn_backbone(cnn_layer)

    if use_attention:
        # Add attention layer.
        attention_layer = AdditiveAttention(name='Attention')([cnn_layer, rnn_layer])
        concatenate_layer = Concatenate(name='Concatenate')([cnn_layer, attention_layer])
        # Connect to the fully connected layer.
        dense_layer = Dense(units=characters_size + 1, activation='softmax', name='Output-Dense')(concatenate_layer)
    else:
        # Connect to the fully connected layer.
        dense_layer = Dense(units=characters_size + 1, activation='softmax', name='Output-Dense')(rnn_layer)

    ctc_layer = CTCLayer(name='CTC-Loss')(input_label, dense_layer)

    if trainable:
        model = Model(inputs=[input_image, input_label], outputs=ctc_layer, name='crnn_model_train')
    else:
        model = Model(inputs=input_image, outputs=dense_layer, name='crnn_model_inference')

    return model


class CTCLayer(Layer):
    """This layer creates a CTC loss function to convert the input tensor into a CTC-encoded tensor.

    Attributes:
        name: str,
            the name of layer.
        loss_fn: function,
            CTC loss function.
    """
    def __init__(self, name=None):
        super(CTCLayer, self).__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        """Implementation of the CTC layer.

        Args:
            y_true: tf.Tensor, containing the truth labels.
            y_pred: tf.Tensor, containing the prediction labels.

        Return:
            The prediction labels tensor.
        """
        batch_length = tf.cast(tf.shape(y_true)[0], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')

        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype='int64')
        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype='int64')

        # Compute loss.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
