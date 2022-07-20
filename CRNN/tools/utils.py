import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend import ctc_decode
from keras.layers import StringLookup

from CRNN import LOGGER


def get_image_format(dataset_dir: Union[str, os.PathLike],
                     language: str) -> str:
    """Get the format of the image.

    Args:
        dataset_dir: str or os.PathLike,
            the path of dataset.
        language: {'en', 'zh'},
            language of the log.

    Return:
        Format of the image.

    Raise:
        ValueError: Image format is not supported.
    """
    image = str(list(Path(dataset_dir).glob('*'))[0])  # Just look at a random one.
    image = image.lower()
    image_format = re.findall(r'\.[^.]+', image)[0][1:]

    if image_format in ('jpg', 'jpeg', 'png'):
        return image_format
    else:
        if language == 'zh':
            LOGGER.error('请检查您图片的格式!')
        else:
            LOGGER.error('Please check the format of your image!')

        raise ValueError('Image format is not supported.')


def get_dataset_summary(dataset_dir: Union[str, os.PathLike],
                        label_path: Union[str, os.PathLike] = None,
                        image_format: str = 'png',
                        fchar: str = '*',
                        inference: bool = False,
                        language: str = 'en',
                        sort: bool = False) -> Union[List[str],
                                                     Tuple[List[str], List[str], int, List[str]]]:
    """Get the summary of the dataset.

    Args:
        dataset_dir: str or os.PathLike,
            the path of dataset.
        label_path: str or os.PathLike, default = None
            csv file containing image labels,
             if None, the image's filename needs to be a label.
        image_format: str, default = 'png',
            the format of image.
        fchar: str, default = '*',
            filled character.
        inference: bool, default = False,
            if True, the model will be trained, else the model will be trained and inferred.
        language: {'en', 'zh'}, default = 'en',
            language of the log.
        sort: bool, default = False,
            if True the images will be sorted.

    Returns:
        Training mode:
            The list of images and labels; the max length of labels; the characters in the dataset.
        Inference mode:
            Only the list of images.
    """
    if label_path is None:
        images, labels, max_length, truncation_length, characters = _from_directory(dataset_dir,
                                                                                    image_format,
                                                                                    fchar,
                                                                                    sort)
    else:
        images, labels, max_length, truncation_length, characters = _from_dataframe(dataset_dir,
                                                                                    label_path,
                                                                                    image_format,
                                                                                    fchar)

    if inference:
        if language == 'zh':
            LOGGER.info(f'一共发现 {len(images)} 张测试图片.')
        else:
            LOGGER.info(f'Number of test images found: {len(images)}.')

        return images
    else:
        if language == 'zh':
            LOGGER.info(f'一共发现 {len(images)} 张图片.')
            LOGGER.info(f'一共发现 {len(labels)} 个标签.')
            LOGGER.info(f'一共发现 {len(characters)} 个字符.')
            LOGGER.info(f'标签的最大长度是 {max_length}.')
            LOGGER.info(f'预处理修剪的长度是 {truncation_length}.')
        else:
            LOGGER.info(f'Number of images found: {len(images)}.')
            LOGGER.info(f'Number of labels found: {len(labels)}.')
            LOGGER.info(f'Number of characters found: {len(characters)}.')
            LOGGER.info(f'Number of max length of labels is: {max_length}.')
            LOGGER.info(f'The clip length is: {truncation_length}.')

        return images, labels, max_length, characters


def decode_predictions(y_pred: np.ndarray,
                       max_length: int,
                       decoder: StringLookup) -> List[str]:
    """Decode prediction data into human-readable labels.

    Args:
        y_pred: np.ndarray,
            the result predicted by the model.
        max_length: int,
            the max length of labels.
        decoder: keras.layers.StringLookup,
            character decoder.

    Return:
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


def _from_directory(dataset_dir: Union[str, os.PathLike],
                    image_format: str,
                    fchar: str,
                    sort: bool) -> Tuple[List[str], List[str], int, int, List[str]]:
    """Get dataset from directory, the filename is the label.

    Args:
        dataset_dir: str or os.PathLike,
            the path of dataset.
        image_format: str,
            the format of image.
        fchar: str,
            filled character.
        sort: bool,
            if True the images will be sorted.

    Returns:
        The list of images and labels;
        the max length of labels;
        the truncation length;
        the characters in the dataset.
    """
    images = list(map(str, list(Path(dataset_dir).glob('*.' + image_format))))
    # Sort images file.
    if sort:
        pattern = re.compile(r'\D+(\d+)\.' + image_format)
        images.sort(key=lambda x: int(re.match(pattern, x).group(1)))

    # Get labels.
    labels = [img.split(os.path.sep)[-1].split('.' + image_format)[0] for img in images]
    labels, max_length, truncation_length, characters = _handling_labels(labels, fchar)

    return images, labels, max_length, truncation_length, characters


def _from_dataframe(dataset_dir: Union[str, os.PathLike],
                    label_path: Union[str, os.PathLike],
                    image_format: str,
                    fchar) -> Tuple[List[str], List[str], int, int, List[str]]:
    """Get dataset from `pd.DataFrame`, the `pd.DataFrame` contains labels for images.

    Args:
        dataset_dir: str or os.PathLike,
            the path of dataset.
        label_path: str or os.PathLike,
            csv file containing image labels,
             if None, the image's filename needs to be a label.
        image_format: str,
            the format of image.
        fchar: str,
            filled character.

    Returns:
        The list of images and labels;
        the max length of labels;
        the truncation length;
        the characters in the dataset.
    """
    images = list(map(str, list(Path(dataset_dir).glob('*.' + image_format))))
    # Sort images file.
    pattern = re.compile(r'\D+(\d+)\.' + image_format)
    images.sort(key=lambda x: int(re.match(pattern, x).group(1)))

    # Get labels.
    labels = list(pd.read_csv(label_path)['label'])
    labels, max_length, truncation_length, characters = _handling_labels(labels, fchar)

    return images, labels, max_length, truncation_length, characters


def _get_truncation_length(label_length_list: List[int],
                           max_length: int) -> int:
    """Adaptively truncate partially ultra-long labels. Experiments
    found that too long labels will affect the convergence of the model.
     If the proportion of the number of too long labels is small,
     the function will adaptively truncate the ultra-long part.

    Args:
        label_length_list: list of int,
            record the length of each label.
        max_length: int,
            the max length of labels.

    Return:
        The suitable length of labels.
    """
    total_labels = len(label_length_list)
    truncation_length = 0

    length_series = pd.Series(label_length_list).value_counts()

    subsets = 0
    for label_length, num_of_labels in length_series.items():
        truncation_length = label_length
        # Calculate the percentage of subsets.
        subsets += num_of_labels
        if subsets / total_labels >= 0.95:
            break

    return truncation_length if max_length - truncation_length >= 3 else max_length


def _modify_label(labels: List[str],
                  truncation_length: int,
                  fchar: str) -> List[str]:
    """Handle labels of undesired lengths. Too long labels
    will be cropped, too short labels will be inserted at
    random positions.

    Args:
        labels: list of str,
            list of original labels.
        truncation_length: int,
            the suitable length of labels.
        fchar: str,
            filled character.

    Return:
        List of processed labels.
    """
    new_labels = []

    for label in labels:
        if len(label) > truncation_length:
            label = label[0: truncation_length]
        elif len(label) < truncation_length:
            label = list(label)
            while len(label) < truncation_length:
                label.insert(random.randint(0, len(label) + 1), fchar)
            label = ''.join(label)

        new_labels.append(label)

    return new_labels


def _handling_labels(labels: List[str],
                     fchar: str) -> Tuple[List[str], int, int, List[str]]:
    """Label handling function.

    Args:
        labels: list of str,
            list of original labels.
        fchar: str,
            filled character.

    Returns:
        The list of labels;
        the max length of labels;
        the truncation length;
        the characters in the dataset.
    """
    label_length_list = []
    for label in labels:
        label_length_list.append(len(label))

    max_length = max(label_length_list)

    # Handle labels of undesired lengths.
    truncation_length = _get_truncation_length(label_length_list, max_length)
    labels = _modify_label(labels, truncation_length, fchar)

    # Get all characters.
    characters = []
    for label in labels:
        characters.extend(list(label))
    characters = sorted(list(set(characters)))

    return labels, max_length, truncation_length, characters
