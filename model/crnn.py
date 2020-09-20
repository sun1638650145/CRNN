import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, AdditiveAttention, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ctc_batch_cost, cast, shape

from .backbone import get_cnn_backbone, get_rnn_backbone


class CTCLayer(Layer):
    """CTC loss layer.

    This layer creates a CTC loss function
    to convert the input tensor into a CTC-encoded tensor.

    Parameters:
        name: str, the name of layer.
    """
    def __init__(self, name=None):
        super(CTCLayer, self).__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        """This is where the layer's logic lives.

        Parameters:
            y_true: tensor, containing the truth labels.
            y_pred: tensor, containing the prediction labels.

        Returns:
            A tensor or list/tuple of tensors.
        """
        batch_length = cast(shape(y_true)[0], dtype='int64')
        label_length = cast(shape(y_true)[1], dtype='int64')
        input_length = cast(shape(y_pred)[1], dtype='int64')

        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype='int64')
        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype='int64')

        # compute loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


def CRNN(img_width,
         img_height,
         img_channels,
         len_characters,
         trainable=True,
         cnn_backbone_name=None,
         rnn_backbone_name=None):
    """Instantiates the CRNN architecture.

    Parameters:
        img_width:
            int, the width of image.
        img_height:
            int, the height of image.
        img_channels:
            int, the channels of image.
        len_characters:
            int, the length of characters.
        trainable:
            bool, default=True
            If true the model will be trained, if false the model will be inferred.
        cnn_backbone_name:
            str, the name of convolution part of CRNN model.
        rnn_backbone_name:
            str, the name of recurrent part of CRNN model.

    Returns:
        A Keras model instance.
    """
    input_image = Input(shape=(img_width, img_height, img_channels), name='Input-Image', dtype='float32')
    input_label = Input(shape=(None, ), name='Input-Label', dtype='float32')

    # CNN backbone
    cnn_backbone = get_cnn_backbone(cnn_backbone_name)
    cnn_layer = cnn_backbone(input_image)

    # RNN backbone
    rnn_backbone = get_rnn_backbone(rnn_backbone_name)
    rnn_layer = rnn_backbone(cnn_layer)

    # Connect to full-connect-layer.
    dense_layer = Dense(units=len_characters + 1, activation='softmax', name='Output-Dense')(rnn_layer)
    ctc_layer = CTCLayer(name='ctc_loss')(input_label, dense_layer)

    if trainable is True:
        model = Model(inputs=[input_image, input_label], outputs=ctc_layer, name='ocr_model_train')
    else:
        model = Model(inputs=input_image, outputs=dense_layer, name='ocr_model_inference')

    return model


def CRNN_Attention(img_width,
                   img_height,
                   img_channels,
                   len_characters,
                   trainable=True,
                   cnn_backbone_name=get_cnn_backbone('resnet_attention'),
                   rnn_backbone_name=None):
    """Instantiate a CRNN architecture with attention mechanism.

    Parameters:
        img_width:
            int, the width of image.
        img_height:
            int, the height of image.
        img_channels:
            int, the channels of image.
        len_characters:
            int, the length of characters.
        trainable:
            bool, default=True
            If true the model will be trained, if false the model will be inferred.
        cnn_backbone_name:
            str, the name of convolution part of CRNN model.
        rnn_backbone_name:
            str, the name of recurrent part of CRNN model.

    Returns:
        A Keras model instance.
    """
    input_image = Input(shape=(img_width, img_height, img_channels), name='Input-Image', dtype='float32')
    input_label = Input(shape=(None,), name='Input-Label', dtype='float32')

    # CNN backbone
    cnn_backbone = get_cnn_backbone(cnn_backbone_name)
    cnn_layer = cnn_backbone(input_image)

    # RNN backbone
    rnn_backbone = get_rnn_backbone(rnn_backbone_name)
    rnn_layer = rnn_backbone(cnn_layer)

    # Add attention
    attention_layer = AdditiveAttention(name='Attention')([cnn_layer, rnn_layer])
    concatenate_layer = Concatenate(name='Concatenate')([cnn_layer, attention_layer])

    # Connect to full-connect-layer.
    dense_layer = Dense(units=len_characters + 1,
                        activation='softmax',
                        name='Output-Dense')(concatenate_layer)
    ctc_layer = CTCLayer(name='ctc_loss')(input_label, dense_layer)

    if trainable is True:
        model = Model(inputs=[input_image, input_label], outputs=ctc_layer, name='ocr_model_train')
    else:
        model = Model(inputs=input_image, outputs=dense_layer, name='ocr_model_inference')

    return model