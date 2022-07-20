from typing import Callable, Optional

from keras.applications import EfficientNetB3
from keras.applications import ResNet50V2
from keras.layers import Layer
from keras.layers import Bidirectional, Dense, Dropout, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM, GRU


def get_cnn_backbone(name: Optional[str]) -> Callable[[Layer], Layer]:
    """Get the convolution backbone in CRNN.

    Args:
        name: {'resnet', 'efficientnet', 'plain_cnn'},
            the name of cnn_backbone.

    Return:
        The convolution net backbone function.
    """
    if name == 'resnet':
        backbone_fn = _resnet50_backbone
    elif name == 'efficientnet':
        backbone_fn = _efficientnet_backbone
    else:
        backbone_fn = _plain_cnn_backbone

    return backbone_fn


def get_rnn_backbone(name: Optional[str]) -> Callable[[Layer], Layer]:
    """Get the recurrent backbone in CRNN.

    Args:
        name: {'bi_gru', 'plain_rnn'},
            the name of rnn_backbone.

    Return:
        The recurrent net backbone function.
    """
    if name == 'bi_gru':
        backbone_fn = _bi_gru_backbone
    else:
        backbone_fn = _plain_rnn_backbone

    return backbone_fn


def _resnet50_backbone(input_layer: Input) -> Layer:
    """ResNet50V2 convolution net backbone.

    Args:
        input_layer: keras.layers.Input,
            the input layer of the model.

    Return:
        The `keras.layers.Layer`.
    """
    layer = ResNet50V2(include_top=False, weights='imagenet')(input_layer)

    # The down sampling factor of ResNet50 is 28,
    # the shape of the ResNet50 output layer is (None, image_width / 28, image_height / 28, 2048).
    layer = Reshape(target_shape=(-1, 512), name='Reshape')(layer)
    layer = Dense(units=128, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.25, name='Dropout')(layer)

    return layer


def _efficientnet_backbone(input_layer: Input) -> Layer:
    """EfficientNetB3 convolution net backbone.

    Args:
        input_layer: keras.layers.Input,
            the input layer of the model.

    Return:
        The `keras.layers.Layer`.
    """
    layer = EfficientNetB3(include_top=False, weights='imagenet')(input_layer)

    # The down sampling factor of EfficientNetB3 is 32,
    # The shape of the EfficientNetB3 output layer is (None, image_width / 32, image_height / 32, 1536).
    layer = Reshape(target_shape=(-1, 384), name='Reshape')(layer)
    layer = Dense(units=128, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.5, name='Dropout')(layer)

    return layer


def _plain_cnn_backbone(input_layer: Input) -> Layer:
    """Plain convolution net backbone.

    Args:
        input_layer: keras.layers.Input,
            the input layer of the model.

    Return:
        The `keras.layers.Layer`.
    """
    layer = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   name='Conv1')(input_layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(layer)

    layer = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   kernel_initializer='he_normal',
                   name='Conv2')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(layer)

    # The down sample factor is 4.
    new_shape = ((input_layer.shape[1] // 4), (input_layer.shape[2] // 4) * 64)
    layer = Reshape(target_shape=new_shape, name='Reshape')(layer)
    layer = Dense(units=128, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.2, name='Dropout')(layer)

    return layer


def _bi_gru_backbone(layer: Layer) -> Layer:
    """Bidirectional GRU recurrent net backbone.

    Args:
        layer:
            keras.layers.Layer, the input convolutional net.

    Return:
        The `keras.layers.Layer`.
    """
    layer = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.25), name='Bi-GRU1')(layer)
    layer = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.25), name='Bi-GRU2')(layer)

    return layer


def _plain_rnn_backbone(layer: Layer) -> Layer:
    """Plain recurrent net backbone.

    Args:
        layer:
            keras.layers.Layer, the input convolutional net.

    Return:
        The `keras.layers.Layer`.
    """
    layer = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.25), name='Bi-LSTM1')(layer)
    layer = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.25), name='Bi-LSTM2')(layer)

    return layer
