import string
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.layers import StringLookup

from CRNN import LOGGER


def character_encoder(vocab: List[str]) -> StringLookup:
    """Character encoder.

    Args:
        vocab: list of str,
            characters to be encoded.

    Return:
        Character encoder.
    """
    char_to_num = StringLookup(max_tokens=None,
                               num_oov_indices=0,
                               mask_token=None,
                               vocabulary=vocab,
                               invert=False)

    return char_to_num


def character_decoder(encoder: StringLookup) -> StringLookup:
    """Character decoder.

    Args:
        encoder: keras.preprocessing.StringLookup,
            character encoder.

    Return:
        Character decoder.
    """
    num_to_char = StringLookup(max_tokens=None,
                               num_oov_indices=1,
                               mask_token=None,
                               vocabulary=encoder.get_vocabulary(),
                               invert=True)

    return num_to_char


def train_validation_split(images: List[str],
                           labels: List[str],
                           validation_size: float = 0.2,
                           shuffle: bool = True,
                           language: str = 'en') -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split datasets into random train and validation subsets.

    Args:
        images: list of str,
            the list of images.
        labels: list of str,
            the list of labels.
        validation_size: float, default=0.2,
            size of validation set.
        shuffle: bool, default=True,
            shuffle the dataset.
        language: {'en', 'zh'}, default='en',
            language of the log.

    Returns:
        Tuple of train dataset and validation dataset.
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    indices = np.arange(images.size)

    # Shuffle the dataset.
    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(images.size * (1 - validation_size))
    if language == 'zh':
        LOGGER.info(f'训练集一共有 {train_samples} 张图片, 验证集一共有 {images.size - train_samples} 张图片.')
    else:
        LOGGER.info(f'Number of train set images: {train_samples},'
                    f' number of validation set images: {images.size - train_samples}')

    x_train, x_validation = images[indices[: train_samples]], images[indices[train_samples:]]
    y_train, y_validation = labels[indices[: train_samples]], labels[indices[train_samples:]]

    return (x_train, y_train), (x_validation, y_validation)


def create_dataset(x: Union[np.ndarray, List[str]],
                   y: Optional[np.ndarray] = None,
                   batch_size: int = 32,
                   image_height: int = 100,
                   image_width: int = 200,
                   image_channels: int = 1,
                   image_format: str = None,
                   encoder: Optional[StringLookup] = None,
                   inference: bool = False,
                   language: str = 'en') -> tf.data.Dataset:
    """Create a `tf.data.Dataset` for training or inference.

    Args:
        x: np.ndarray or list of str,
            image data.
        y: np.ndarray, default=None,
            label data, if inference is True, y is None.
        batch_size: int, default=32,
            number of samples per gradient update.
        image_height: int, default=100,
            the height of image.
        image_width: int, default=200,
            the width of image.
        image_channels: {1, 3}, default=1,
            the channels of image.
        image_format: str,
            the format of image.
        encoder: keras.preprocessing.StringLookup,
            character encoder.
        inference: bool, default=False,
            if True, the model will be trained, else the model will be trained and inferred.
        language: {'en', 'zh'}, default='en',
            language of the log.

    Return:
        A `tf.data.Dataset`.
    """
    # Get the default encoder.
    if encoder is None:
        encoder = character_encoder(list(string.digits + string.ascii_letters))

    if inference:
        # Inference mode does not require labeled data.
        def _map_func(image):
            return _preprocessing_single_sample(image_path=image,
                                                label=None,
                                                image_height=image_height,
                                                image_width=image_width,
                                                image_channels=image_channels,
                                                image_format=image_format,
                                                encoder=encoder,
                                                inference=inference,
                                                language=language)
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        def _map_func(image, label):
            return _preprocessing_single_sample(image_path=image,
                                                label=label,
                                                image_height=image_height,
                                                image_width=image_width,
                                                image_channels=image_channels,
                                                image_format=image_format,
                                                encoder=encoder,
                                                inference=inference,
                                                language=language)
        # _map_func = lambda image, label: _preprocessing_single_sample(image_path=image,
        #                                                               label=label,
        #                                                               image_height=image_height,
        #                                                               image_width=image_width,
        #                                                               image_channels=image_channels,
        #                                                               image_format=image_format,
        #                                                               encoder=encoder,
        #                                                               inference=inference,
        #                                                               language=language)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # The use of parallelization strategies can
    # be shown to significantly improve performance.
    dataset = (
        dataset
        .map(map_func=_map_func, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset


def _preprocessing_single_sample(image_path: tf.Tensor,
                                 label: Optional[tf.Tensor] = None,
                                 image_height: int = 100,
                                 image_width: int = 200,
                                 image_channels: int = 1,
                                 image_format: str = None,
                                 encoder: Optional[StringLookup] = None,
                                 inference: bool = False,
                                 language: str = 'en') -> Union[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Preprocessing single sample, convert single sample to tf.Tensor.

    Args:
        image_path: tf.Tensor,
            a string tensor of image paths.
        label:
            the label of image.
        image_height: int, default=100,
            the height of image.
        image_width: int, default=200,
            the width of image.
        image_channels: {1, 3}, default=1,
            the channels of image.
        image_format: str,
            the format of image.
        encoder: keras.preprocessing.StringLookup,
            character encoder.
        inference: bool, default=False,
            if True, the model will be trained, else the model will be trained and inferred.
        language: {'en', 'zh'}, default='en',
            language of the log.

    Returns:
        Training mode:
            The dict of image and label tensor.
        Inference mode:
            Only the dict of image tensor.

    Raise:
        ValueError: Image format is not supported.
    """
    image = tf.io.read_file(image_path)

    if image_format == 'png':
        image = tf.image.decode_png(image, channels=image_channels)
    elif image_format in ('jpg', 'jpeg'):
        image = tf.image.decode_jpeg(image, channels=image_channels)
    else:
        if language == 'zh':
            LOGGER.error('请检查您图片的格式!')
        else:
            LOGGER.error('Please check the format of your image!')

        raise ValueError('Image format is not supported.')

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_height, image_width])
    # Convert image width and height.
    image = tf.transpose(image, perm=[1, 0, 2])

    if inference:
        return {'Input-Image': image}
    else:
        label = encoder(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        return {'Input-Image': image, 'Input-Label': label}
