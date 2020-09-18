import string
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='CRNN')


def character_encoder(vocab):
    """Character encoder

    Parameters:
        vocab: list, characters to be encoded.

    Returns:
        Character encoder(keras.preprocessing.StringLookup).
    """
    char_to_num = StringLookup(mask_token=None,
                               num_oov_indices=0,
                               vocabulary=list(vocab),
                               invert=False)

    return char_to_num


def character_decoder(encoder):
    """Character decoder

    Parameters:
        encoder: keras.preprocessing.StringLookup, character encoder.

    Returns:
        Character decoder(keras.preprocessing.StringLookup).
    """
    num_to_char = StringLookup(mask_token=None,
                               num_oov_indices=1,
                               vocabulary=encoder.get_vocabulary(),
                               invert=True)

    return num_to_char


def train_validation_split(images, labels, validation_size=0.2, shuffle=True, language='en'):
    """Split datasets into random train and validation subsets.

    Parameters:
        images: array-like
            index of image files.
        labels: array-like
            index of labels.
        validation_size: float, default=0.2
            size of validation set.
        shuffle: bool, default=True
            whether or not to shuffle the dataset before splitting.
        language:
            str, language of the log.

    Returns:
        tuple of train dataset(images, labels) and validation dataset(images, labels)
    """
    images = np.asarray(images)
    labels = np.asarray(labels)

    size = len(images)
    indices = np.arange(size)

    # shuffle the dataset
    if shuffle is True:
        np.random.shuffle(indices)

    train_samples = int(size * (1 - validation_size))
    if language is 'en':
        logger.info('Number of train set images: {}'.format(train_samples))
        logger.info('Number of validation set images: {}'.format(size - train_samples))
    elif language is 'zh':
        logger.info('训练集一共有 {} 张图片'.format(train_samples))
        logger.info('验证集一共有 {} 张图片'.format(size - train_samples))

    x_train, x_validation = images[indices[: train_samples]], images[indices[train_samples:]]
    y_train, y_validation = labels[indices[: train_samples]], labels[indices[train_samples:]]

    return (x_train, y_train), (x_validation, y_validation)


def _preprocessing_single_sample(img_path,
                                 label,
                                 img_height,
                                 img_width,
                                 img_channels,
                                 encoder,
                                 image_format,
                                 language,
                                 inference_status):
    """Preprocessing Single Sample,
    convert single sample to TensorFlow Tensor.

    Parameters:
        img_path:
            image files.
        label:
            the label of image.
        img_height:
            int, the height of image.
        img_width:
            int, the width of image.
        img_channels:
            int, the channels of image.
        encoder:
            character encoder.
        image_format:
            str, the format of images.
        language:
            str, language of the log.
        inference_status:
            bool, default=True
            If true the model will be trained, else the model will be inferred.

    Returns:
        If it is not inference mode, a dict of image and label tensors;
        else a dict of one image tensor.
    """
    img = tf.io.read_file(img_path)

    if image_format is 'png':
        img = tf.image.decode_png(img, channels=img_channels)
    elif image_format in ('jpg', 'jpeg'):
        img = tf.image.decode_jpeg(img, channels=img_channels)
    else:
        if language is 'zh':
            error_info = '请检查您图片的格式'
        else:
            error_info = 'Please check the format of your image'
        logger.error(error_info)
        raise ValueError

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    # Convert image width and height.
    img = tf.transpose(img, perm=[1, 0, 2])

    if inference_status is False:
        label = encoder(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        return {'image': img, 'label': label}
    else:
        return {'image': img}


def create_tf_dataset(x,
                      y=None,
                      batch_size=32,
                      img_height=100,
                      img_width=200,
                      img_channels=1,
                      encoder=None,
                      image_format='png',
                      language='en',
                      inference_status=False):
    """Create a tf.data.Dataset for training or inference.

    Args:
        x: Array-like,
            Input images list.
        y: Array-like, default=None
            If inference status is False, y is not None.
        batch_size: Int, default=32.
            Number of samples per gradient update.
        img_height:
            int, the height of image.
        img_width:
            int, the width of image.
        img_channels:
            int, the channels of image.
        encoder:
            character encoder.
        image_format:
            str, the format of images.
        language:
            str, language of the log.
        inference_status:
            bool, default=True
            If true the model will be trained, else the model will be inferred.

    Returns:
        tf.data.Dataset
    """
    if encoder is None:
        encoder = character_encoder(list(string.digits + string.ascii_letters))

    if inference_status is False:
        def map_func(img, label):
            return _preprocessing_single_sample(img,
                                                label,
                                                img_height,
                                                img_width,
                                                img_channels,
                                                encoder,
                                                image_format,
                                                language,
                                                inference_status)
        # map_func = lambda img, label: _preprocessing_single_sample(img,
        #                                                            label,
        #                                                            img_height,
        #                                                            img_width,
        #                                                            img_channels,
        #                                                            encoder,
        #                                                            image_format,
        #                                                            language,
        #                                                            inference_status)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    else:
        def map_func(img):
            return _preprocessing_single_sample(img,
                                                None,
                                                img_height,
                                                img_width,
                                                img_channels,
                                                encoder,
                                                image_format,
                                                language,
                                                inference_status)
        dataset = tf.data.Dataset.from_tensor_slices(x)

    # The use of parallelization strategies can
    # be shown to significantly improve performance.
    dataset = (
        dataset
        .map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset