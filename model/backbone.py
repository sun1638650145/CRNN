from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Reshape, Dense, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.efficientnet import EfficientNetB3


def get_cnn_backbone(backbone_name):
    """Get the convolution backbone in CRNN.

    Parameters:
        backbone_name: str, the name of cnn_backbone.

    Returns:
        cnn_backbone function
    """
    if backbone_name is 'resnet':
        cnn_backbone_fn = _resnet50_cnn_backbone
    elif backbone_name is 'efficientnet':
        cnn_backbone_fn = _efficientnetb3_cnn_backbone
    else:
        cnn_backbone_fn = _plain_cnn_backbone

    return cnn_backbone_fn


def _plain_cnn_backbone(input_layer):
    """Plain convolution part of CRNN model.

    Parameters:
        input_layer:
            Keras Layer, model input.

    Returns:
        Keras Layer.
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

    # downsample factor is 4
    new_shape = ((input_layer.shape[1] // 4), (input_layer.shape[2] // 4) * 64)
    layer = Reshape(target_shape=new_shape, name='Reshape')(layer)
    layer = Dense(units=64, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.2, name='Dropout')(layer)

    return layer


def _resnet50_cnn_backbone(input_layer):
    """ResNet50V2 convolution part of CRNN model.

    Parameters:
        input_layer:
            Keras Layer, model input.

    Returns:
        Keras Layer.
    """
    layer = ResNet50V2(include_top=False, weights='imagenet')(input_layer)

    # The down sampling factor of ResNet50 is 28.
    # The shape of the ResNet50 output layer is (None, img_width / 28, img_height / 28, 2048).

    layer = Reshape(target_shape=(-1, 512), name='Reshape')(layer)
    layer = Dense(units=128, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.25, name='Dropout')(layer)

    return layer


def _efficientnetb3_cnn_backbone(input_layer):
    """EfficientNetB3 convolution part of CRNN model.

        Parameters:
            input_layer:
                Keras Layer, model input.

        Returns:
            Keras Layer.
        """
    layer = EfficientNetB3(include_top=False, weights='imagenet')(input_layer)

    # The down sampling factor of EfficientNetB3 is 32.
    # The shape of the EfficientNetB3 output layer is (None, img_width / 32, img_height / 32, 1536).

    layer = Reshape(target_shape=(-1, 384), name='Reshape')(layer)
    layer = Dense(units=128, activation='relu', name='Dense1')(layer)
    layer = Dropout(rate=0.5, name='Dropout')(layer)

    return layer


def get_rnn_backbone(backbone_name):
    """Get the recurrent backbone in CRNN.

    Parameters:
        backbone_name: str, the name of rnn_backbone.

    Returns:
        rnn_backbone function
    """
    if backbone_name is 'bi_gru':
        rnn_backbone_fn = _bi_gru_backbone
    else:
        rnn_backbone_fn = _plain_rnn_backbone

    return rnn_backbone_fn


def _plain_rnn_backbone(input_layer):
    """Plain recurrent part of CRNN model.

    Parameters:
        input_layer:
            Keras Layer, model input.

    Returns:
        Keras Layer.
    """

    layer = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.25), name='Bi-LSTM1')(input_layer)
    layer = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.25), name='Bi-LSTM2')(layer)

    return layer


def _bi_gru_backbone(input_layer):
    """GRU recurrent part of CRNN model.

        Parameters:
            input_layer:
                Keras Layer, model input.

        Returns:
            Keras Layer.
        """

    layer = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.25), name='Bi-GRU1')(input_layer)
    layer = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.25), name='Bi-GRU2')(layer)

    return layer