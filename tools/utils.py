import os
import re
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.backend import ctc_decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='CRNN')


def get_image_format(dataset_dir, language):
    """Get image format.

    Parameters:
        dataset_dir: dataset_dir: str or path
            The path of dataset.
        language:
            str, language of the log.

    Returns:
        Image format.
    """
    images = list(map(str, list(Path(dataset_dir).glob('*'))))
    images_format = re.findall(r'\.[^.]+', images[0])
    if images_format[0] == '.jpg':
        return 'jpg'
    elif images_format[0] == '.png':
        return 'png'
    else:
        if language is 'zh':
            error_info = '请检查您图片的格式'
        else:
            error_info = 'Please check the format of your image'
        logger.error(error_info)
        raise ValueError


def _get_clip_length(label_length_list, max_length):
    """Experiments have found that too long labels
    will affect the convergence of the model.
    If the proportion of too long labels is small,
    the function will adaptively truncate the ultra-long part.

    Args:
        label_length_list: list,
            record the length of each label.
        max_length: int,
            the max length of labels.
    Returns:
        The suitable length of labels.
    """
    clip_length = 0

    length_series = pd.DataFrame(label_length_list).value_counts()
    length_dataframe = pd.DataFrame(length_series)

    num_of_labels = sum(length_dataframe[0])
    sub_set = 0
    for label_length, num_of_label in length_dataframe.iterrows():
        clip_length = label_length[0]
        sub_set += int(num_of_label)

        if sub_set / num_of_labels >= 0.95:
            break

    if max_length - clip_length >= 3:
        return clip_length
    else:
        return max_length


def _fill_label(labels, clip_length, filled_char):
    """Fill in label with insufficient length.

    Parameters:
        labels: list, labels list.
        clip_length: int,
            the length of the label being clip.
        filled_char: char, filled character.

    Returns:
        filled labels list.
    """
    new_labels = []
    for label in labels:
        label = str(label)
        if len(label) < clip_length:
            label = list(label)
            while len(label) < clip_length:
                label.insert(random.randint(0, len(label) + 1), filled_char)
            label = ''.join(label)
            new_labels.append(label)
        elif len(label) > clip_length:
            label = label[0:clip_length]
            new_labels.append(label)
        else:
            new_labels.append(label)

    return new_labels


def _from_directory(dataset_dir, image_format, sort, filled_char):
    """Get dataset from directory.

    Parameters:
        dataset_dir: str or path
            The path of dataset.
        image_format:
            str, the format of images.
        sort:
            bool, if true the model will be sorted.
        filled_char: char, filled character.

    Returns:
        The number of images, labels, and characters in the dataset,
        and the max length of labels.
    """
    images = list(map(str, list(Path(dataset_dir).glob('*.' + image_format))))

    # sort images file
    if sort is True:
        pattern = re.compile('\\D+(\\d+)\\.' + image_format)
        images.sort(key=lambda x: int(re.match(pattern, x).group(1)))

    labels = [img.split(os.path.sep)[-1].split('.' + image_format)[0] for img in images]

    label_length_list = []
    for label in labels:
        label = str(label)
        label_length_list.append(len(label))
    max_length = max(label_length_list)

    clip_length = _get_clip_length(label_length_list, max_length)

    # fill in label with insufficient length
    labels = _fill_label(labels, clip_length, filled_char)

    characters = set(char for label in labels for char in label)
    characters = sorted(list(characters))

    return images, labels, characters, max_length, clip_length


def _from_dataframe(dataset_dir, csv_path, image_format, filled_char):
    """Get dataset from csv file.

    Parameters:
        dataset_dir: str or path
            the path of dataset.
        csv_path: csv file
            csv containing the labels of the images,
            if None, the filename of the image needs to be a label.
        image_format:
            str, the format of images.
        filled_char: char, filled character.

    Returns:
        the number of images, labels, and characters in the dataset,
        and the max length of labels.
    """
    images = list(map(str, list(Path(dataset_dir).glob('*.' + image_format))))

    # sort images file
    pattern = re.compile('\\D+(\\d+)\\.' + image_format)
    images.sort(key=lambda x: int(re.match(pattern, x).group(1)))

    labels_dataframe = pd.read_csv(csv_path)
    labels = list(labels_dataframe['label'])

    label_length_list = []
    for label in labels:
        label_length_list.append(len(str(label)))
    max_length = max(label_length_list)

    clip_length = _get_clip_length(label_length_list, max_length)

    # fill in label with insufficient length
    labels = _fill_label(labels, clip_length, filled_char)

    characters = set(char for label in labels for char in label)
    characters = sorted(list(characters))

    return images, labels, characters, max_length, clip_length


def get_dataset_summary(dataset_dir,
                        csv_path=None,
                        image_format='png',
                        language='en',
                        inference_status=False,
                        sort=False,
                        filled_char='-'):
    """get the summary of the dataset.

    Parameters:
        dataset_dir: str or path
            the path of dataset.
        csv_path: csv file
            csv containing the labels of the images,
            if None, the filename of the image needs to be a label.
        image_format:
            str, the format of images.
        language:
            str, language of the log.
        inference_status:
            bool, default=True
            If true the model will be trained, else the model will be inferred.
        sort:
            bool, default=False,
            if true the model will be sorted.
        filled_char: str, default='-',
            filled character.

    Returns:
        the number of images, labels, and characters in the dataset, and the max length of labels.
    """
    if csv_path is None:
        images, labels, characters, max_length, clip_length = _from_directory(dataset_dir,
                                                                              image_format,
                                                                              sort,
                                                                              filled_char)
    else:
        images, labels, characters, max_length, clip_length = _from_dataframe(dataset_dir,
                                                                              csv_path,
                                                                              image_format,
                                                                              filled_char)

    if inference_status is False:
        if language is 'en':
            logger.info('Number of images found: {}'.format(len(images)))
            logger.info('Number of labels found: {}'.format(len(labels)))
            logger.info('Number of characters found: {}'.format(len(characters)))
            logger.info('Number of max length of labels is: {}'.format(max_length))
            logger.info('The clip length is: {}'.format(clip_length))
        elif language is 'zh':
            logger.info('一共发现 {} 张图片'.format(len(images)))
            logger.info('一共发现 {} 个标签'.format(len(labels)))
            logger.info('一共发现 {} 个字符'.format(len(characters)))
            logger.info('标签的最大长度是 {}'.format(max_length))
            logger.info('预处理修剪的长度是 {}'.format(clip_length))

        return images, labels, characters, max_length
    else:
        if language is 'en':
            logger.info('Number of images found: {}'.format(len(images)))
        elif language is 'zh':
            logger.info('一共发现 {} 张图片'.format(len(images)))

        return images


def visualize_train_data(dataset, decoder):
    """Visualize the train data.

    Parameters:
        dataset: tf.data.Dataset,
            dataset that needs to be visualized.
        decoder: character decoder.

    Notes:
        Don’t worry if you find a dash on the label,
        this is automatically filled in when preprocessing the data.
    """
    _, ax = plt.subplots(4, 4, figsize=(10, 5))

    for batch in dataset.take(1):
        images = batch['image']
        labels = batch['label']
        for i in range(16):
            img = (images[i] * 255).numpy().astype('uint8')
            label = tf.strings.reduce_join(decoder(labels[i])).numpy().decode('utf-8')

            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap='Greys')
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis('off')

    plt.show()


def decode_predictions(y_pred, max_length, decoder):
    """Decode prediction data into human-readable labels.

    Parameters:
        y_pred: Tensor,
            tensor predicted by the model.
        max_length: int,
            the max length of label
        decoder: character decoder.

    Returns:
        List of human-readable labels.
    """
    input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]

    # We only use the first tensor in the tuple.
    results = ctc_decode(y_pred, input_length=input_length, greedy=True)[0][0][:, :max_length]

    decode_text = []
    for result in results:
        result = tf.strings.reduce_join(decoder(result)).numpy().decode('utf-8')
        decode_text.append(result)

    return decode_text