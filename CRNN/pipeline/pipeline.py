import os
from typing import Optional, Tuple, Union

import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam

from CRNN import LOGGER
from CRNN.model import CRNN
from CRNN.tools import character_decoder
from CRNN.tools import character_encoder
from CRNN.tools import create_dataset
from CRNN.tools import decode_predictions
from CRNN.tools import get_callback
from CRNN.tools import get_dataset_summary
from CRNN.tools import get_image_format
from CRNN.tools import train_validation_split


class CRNNPipeline(object):
    """CRNN pipeline.
    You only need to configure a few parameters(if you don't want to configure,
     you can use the default parameters) and specify the path of the dataset,
     the CRNN pipeline can fit automatically.

    Attributes:
        train_dataset_dir: str or os.PathLike,
            the path of train dataset.
        test_dataset_dir: str or os.PathLike, default=None,
            the path of tset dataset; the file name must be created
            according to the serial number, such as "1.jpg, 2.jpg, ...".
        train_label_path: str or os.PathLike, default=None,
            csv file containing image labels,
             if None, the image's filename needs to be a label.
        checkpoints_dir: str os.PathLike, default='./checkpoint/',
            the location where checkpoints are saved.
        validation_size: float, default=0.2,
            size of validation set.
        batch_size: int, default=64,
            number of samples per gradient update.
        learning_rate: float, default=1e-3,
            the learning rate of the optimizer.
        epochs: int, default=50,
            number of epochs to train the model.
        multi_gpus: bool, default=False,
            use multi-GPU parallelism (make sure you have more than two physical GPUs).
        cnn_backbone: {'resnet', 'efficientnet', 'plain_cnn'}, default='plain_cnn',
            the name of convolution part of CRNN model.
        rnn_backbone: {'bi_gru', 'plain_rnn'}, default='plain_rnn',
            the name of recurrent part of CRNN model.
        use_attention: bool, default=False,
            the model has attention layers.
        image_height: int, default=50,
            the height of image.
        image_width: int, default=200,
            the width of image.
        image_channels: {1, 3}, default=1,
            the channels of image.
        image_format: str, default=None,
            the format of image,
             if None, it will be automatically obtained.
        use_tensorboard: bool, default=True,
            you can view some useful information on the TensorBoard.
        fchar: str, default='*',
            filled character.
        inference: bool, default=False,
            if True, the model will be trained, else the model will be trained and inferred.
        language: {'en', 'zh'}, default='en',
            language of the log.
        max_length: int,
            the max length of labels.
        characters: list of str,
            the characters in the dataset.
        n2c: keras.layers.StringLookup,
            character decoder.
    """
    def __init__(self,
                 train_dataset_dir: Union[str, os.PathLike],
                 test_dataset_dir: Union[str, os.PathLike] = None,
                 train_label_path: Union[str, os.PathLike] = None,
                 checkpoints_dir: Union[str, os.PathLike] = './results/checkpoint/',
                 validation_size: float = 0.2,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 epochs: int = 50,
                 multi_gpus: bool = False,
                 cnn_backbone: Optional[str] = None,
                 rnn_backbone: Optional[str] = None,
                 use_attention: bool = False,
                 image_height: int = 50,
                 image_width: int = 200,
                 image_channels: int = 3,
                 image_format: Optional[str] = None,
                 use_tensorboard: bool = True,
                 fchar: str = '*',
                 inference: bool = False,
                 language: str = 'en'):
        super(CRNNPipeline, self).__init__()

        # Path parameters.
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.train_label_path = train_label_path
        self.checkpoints_dir = checkpoints_dir

        # Hyper parameters.
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.multi_gpus = multi_gpus
        self.cnn_backbone = cnn_backbone
        self.rnn_backbone = rnn_backbone
        self.use_attention = use_attention

        # Image parameters.
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        if image_format is None:
            self.image_format = get_image_format(train_dataset_dir, language)
        else:
            self.image_format = image_format

        # Other parameters.
        self.use_tensorboard = use_tensorboard
        self.fchar = fchar
        self.inference = inference
        self.language = language

        # Built-in parameters.
        self.max_length = -1
        self.characters = None
        self.n2c = None

    def run(self):
        """Run pipeline."""
        # Load datasets.
        if self.inference:
            train_dataset, val_dataset, test_dataset = self.load_datasets()
        else:
            train_dataset, val_dataset = self.load_datasets()
        if self.language == 'zh':
            LOGGER.info('数据集预处理完成, 即将开始训练...')
        else:
            LOGGER.info('Dataset preprocessing is complete, training will start soon...')

        # Instantiates model.
        if self.inference:
            train_model, inference_model = self._create_model(len(self.characters), self.multi_gpus, self.inference)
            train_model.summary()
            inference_model.summary()
        else:
            train_model = self._create_model(len(self.characters), self.multi_gpus,  self.inference)
            train_model.summary()
        if self.language == 'zh':
            LOGGER.info('模型实例化完成, 开始训练...')
        else:
            LOGGER.info('Model instantiation is complete, start training...')

        # Fit model.
        train_model.fit(x=train_dataset,
                        epochs=self.epochs,
                        verbose=1,
                        callbacks=get_callback(self.checkpoints_dir, self.use_tensorboard),
                        validation_data=val_dataset)

        # Save model.
        train_model.save(filepath='./results/model.h5')
        if self.language == 'zh':
            LOGGER.info('模型训练完成, 并完成保存.')
        else:
            LOGGER.info('The model training is complete, and the save is complete.')

        # Use the model for inference.
        if self.inference:
            if self.language == 'zh':
                LOGGER.info('正在执行推理并将结果保存到文件...')
            else:
                LOGGER.info('Performing inference and saving results to file...')

            inference_model.load_weights('./results/model.h5')
            preds = inference_model.predict(x=test_dataset, verbose=1)
            # Decode to human-readable labels.
            preds = decode_predictions(preds, self.max_length, self.n2c)

            # Handle padding characters and [UNK].
            final_data = []
            for i in range(len(preds)):
                text = preds[i].replace('[UNK]', '')
                text = text.replace(self.fchar, '')
                final_data.append(text)

            # Save to CSV file.
            dataframe = pd.DataFrame(final_data)
            dataframe.to_csv('./results/inference.csv', index=True, header=False, encoding='utf-8')

        if self.language == 'zh':
            LOGGER.info('流水线完成.')
        else:
            LOGGER.info('Pipeline finished.')

    def load_datasets(self) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset],
                                     Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
        """Load the dataset (including preprocessing).

        Returns:
           Datasets for training, validation and testing (probably none).

        Raise:
            FileExistsError: There is no validation dataset.
        """
        images, labels, self.max_length, self.characters = get_dataset_summary(dataset_dir=self.train_dataset_dir,
                                                                               label_path=self.train_label_path,
                                                                               image_format=self.image_format,
                                                                               fchar=self.fchar,
                                                                               inference=False,
                                                                               language=self.language,
                                                                               sort=False)

        # Encode and decode.
        c2n = character_encoder(self.characters)
        self.n2c = character_decoder(c2n)

        # Split datasets.
        (x_train, y_train), (x_val, y_val) = train_validation_split(images,
                                                                    labels,
                                                                    self.validation_size,
                                                                    language=self.language)

        # Create `tf.data.Dataset()`.
        train_dataset = create_dataset(x=x_train,
                                       y=y_train,
                                       batch_size=self.batch_size,
                                       image_height=self.image_height,
                                       image_width=self.image_width,
                                       image_channels=self.image_channels,
                                       image_format=self.image_format,
                                       encoder=c2n,
                                       inference=False,
                                       language=self.language)
        val_dataset = create_dataset(x=x_val,
                                     y=y_val,
                                     batch_size=self.batch_size,
                                     image_height=self.image_height,
                                     image_width=self.image_width,
                                     image_channels=self.image_channels,
                                     image_format=self.image_format,
                                     encoder=c2n,
                                     inference=False,
                                     language=self.language)

        if self.inference:
            if not self.test_dataset_dir:
                if self.language == 'zh':
                    LOGGER.error('请设置验证数据集!')
                else:
                    LOGGER.error('Please set a validation dataset!')

                raise FileExistsError('There is no validation dataset.')

            x_test = get_dataset_summary(dataset_dir=self.test_dataset_dir,
                                         label_path=None,
                                         image_format=self.image_format,
                                         fchar=self.fchar,
                                         inference=self.inference,
                                         language=self.language,
                                         sort=True)

            test_dataset = create_dataset(x=x_test,
                                          y=None,
                                          batch_size=self.batch_size,
                                          image_height=self.image_height,
                                          image_width=self.image_width,
                                          image_channels=self.image_channels,
                                          image_format=self.image_format,
                                          encoder=c2n,
                                          inference=True)

            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, val_dataset

    def _create_model(self,
                      characters_size: int,
                      multi_gpus: bool = False,
                      inference: bool = False) -> Union[Tuple[Model, Model], Model]:
        """Create model.

        Args:
            characters_size: int,
                The size of the characters in the dataset.
            multi_gpus: bool, default=False,
                use multi-GPU parallelism (make sure you have more than two physical GPUs).
            inference: bool, default=False,
                if True, the model will be trained, else the model will be trained and inferred.

        Returns:
            Tht train model and inference model(probably none).
        """
        if multi_gpus:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                train_model = CRNN(self.image_width,
                                   self.image_height,
                                   self.image_channels,
                                   self.cnn_backbone,
                                   self.rnn_backbone,
                                   characters_size,
                                   True,
                                   self.use_attention)
                train_model.compile(optimizer=Adam(self.learning_rate))
        else:
            train_model = CRNN(self.image_width,
                               self.image_height,
                               self.image_channels,
                               self.cnn_backbone,
                               self.rnn_backbone,
                               characters_size,
                               True,
                               self.use_attention)
            train_model.compile(optimizer=Adam(self.learning_rate))

        if inference:
            inference_model = CRNN(self.image_width,
                                   self.image_height,
                                   self.image_channels,
                                   self.cnn_backbone,
                                   self.rnn_backbone,
                                   characters_size,
                                   trainable=False,
                                   use_attention=self.use_attention)

            return train_model, inference_model
        else:
            return train_model
