import logging

import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from ..tools import *
from ..model import CRNN
from .callbacks import get_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='CRNN')


class CRNNPipeline:
    """CRNN pipeline.
    You only need to configure a few parameters(If you don't want to configure,
     you can use the default parameters.) and specify the path of the dataset,
     the CRNN pipeline can fit automatically.

    Parameters:
        train_dataset_dir: str or path,
            the path of train dataset.
        test_dataset_dir: str or path, default=None
            the path of test dataset,
             the file name must be created according to the serial number,
             such as "1.jpg, 2.jpg, ...".
        csv_path: csv file, default=None
            csv containing the labels of the images,
             if None, the filename of the image needs to be a label.
        model_savedpath: str or path,
            H5 file for saving the model.
        checkpoint_dir: str or None, default='./checkpoint/',
            If the value is not None, the checkpoint will be saved in the specified path.
        save_inference: str or None,
            Save the inference result as a csv file.
        image_height: int, default=50,
            the height of image.
        image_width: int, default=200,
            the width of image.
        image_channels: int, default=1,
            the channels of image.
        image_format: str, default=None,
            The format of images,
             if the value is None, it will be automatically obtained.
        cnn_backbone:
            str, the name of convolution part of CRNN model.
        rnn_backbone:
            str, the name of recurrent part of CRNN model.
        batch_size: int, default=32.
            Number of samples per gradient update.
        learning_rate: float, default=1e-3,
            A CRNN parameters.
        epochs: int, default=50,
            Number of epochs to train the model.
        gpus: int, default=None,
            If the value is None, it will be use one device(one CPU or one GPU),
             if the value greater than 2, it will be use multi-GPU device.
        tensorboard: bool, default=True,
            You can view some useful information on the localhost.
        filled_char: str, default='*',
            filled character.
        language: str, default='en',
            language of the log.
        inference_status: bool, default=True
            If true the model will be trained,
             else the model will be trained and inferred.
    """
    def __init__(self,
                 train_dataset_dir,
                 test_dataset_dir=None,
                 csv_path=None,
                 model_savedpath='./model.h5',
                 checkpoint_dir='./checkpoint/',
                 save_inference=None,
                 image_height=50,
                 image_width=200,
                 image_channels=1,
                 image_format=None,
                 cnn_backbone=None,
                 rnn_backbone=None,
                 batch_size=64,
                 learning_rate=1e-3,
                 epochs=50,
                 gpus=None,
                 tensorboard=True,
                 filled_char='*',
                 language='en',
                 inference_status=False):
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.csv_path = csv_path
        self.model_savedpath = model_savedpath
        self.checkpoint_dir = checkpoint_dir
        self.save_inference = save_inference
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.cnn_backbone = cnn_backbone
        self.rnn_backbone = rnn_backbone
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        if gpus is None:
            self.gpus = 0
        else:
            self.gpus = gpus
        self.tensorboard = tensorboard
        self.filled_char = filled_char
        self.language = language
        if image_format is None:
            self.image_format = get_image_format(self.train_dataset_dir, self.language)
        else:
            self.image_format = image_format
        self.inference_status = inference_status

        self.characters = None

    def run(self):
        """Run pipeline."""
        if self.inference_status is True:
            train_tf_dataset, validation_tf_dataset, test_tf_dataset = self._preprocessing()
        else:
            train_tf_dataset, validation_tf_dataset = self._preprocessing()
        if self.language == 'zh':
            logger.info('数据集预处理完成, 即将开始训练.')
        else:
            logger.info('Dataset preprocessing is complete, training will start soon.')
        # Instantiates model
        if self.inference_status is True:
            train_model, inference_model = self._create_model(self.inference_status, self.gpus)
            train_model.summary()
            inference_model.summary()
        else:
            train_model = self._create_model(self.inference_status, self.gpus)
            train_model.summary()
        if self.language == 'zh':
            logger.info('模型实例化完成.')
        else:
            logger.info('Model instantiation completed.')
        # Fit model
        train_model.fit(
            x=train_tf_dataset,
            epochs=self.epochs,
            verbose=1,
            callbacks=get_callback(self.checkpoint_dir, self.tensorboard),
            validation_data=validation_tf_dataset,
        )
        # Save model
        train_model.save(self.model_savedpath)
        if self.language == 'zh':
            logger.info('模型训练完成, 并完成保存.')
        else:
            logger.info('The model training is complete, and the save is complete.')
        # Model inference
        if self.inference_status is True:
            try:
                inference_model.load_weights(self.model_savedpath)
            except:
                if self.language == 'zh':
                    logger.info('模型文件不存在.')
                else:
                    logger.info('Model file does not exist.')
                raise FileNotFoundError
            preds = inference_model.predict(x=test_tf_dataset, verbose=1)
            pred_texts = decode_predictions(preds, self.max_length, self.num_to_char)
            ans = []
            for i in range(len(pred_texts)):
                text = pred_texts[i].replace('[UNK]', '')
                text = text.replace(self.filled_char, '')
                print(i, text)
                ans.append(text)
            if self.save_inference is not None:
                # Save to CSV file.
                dataframe = pd.DataFrame(ans)
                dataframe.to_csv(self.save_inference, index=True, header=False, encoding='utf-8')
        if self.language == 'zh':
            logger.info('流水线完成.')
        else:
            logger.info('Pipeline finished.')

    def _preprocessing(self):
        """Preprocessing dataset.

        Returns:
           Create a tf.data.Dataset for training.
        """
        images, labels, self.characters, self.max_length = get_dataset_summary(self.train_dataset_dir,
                                                                               self.csv_path,
                                                                               self.image_format,
                                                                               self.language,
                                                                               False,
                                                                               False,
                                                                               self.filled_char)
        # encode and decode
        self.char_to_num = character_encoder(self.characters)
        self.num_to_char = character_decoder(self.char_to_num)
        # split datasets
        (x_train, y_train), (x_valid, y_valid) = train_validation_split(images, labels, 0.2)
        # create tf.dataset
        train_dataset = create_tf_dataset(x_train,
                                          y_train,
                                          self.batch_size,
                                          self.image_height,
                                          self.image_width,
                                          self.image_channels,
                                          self.char_to_num,
                                          self.image_format)
        validation_dataset = create_tf_dataset(x_valid,
                                               y_valid,
                                               self.batch_size,
                                               self.image_height,
                                               self.image_width,
                                               self.image_channels,
                                               self.char_to_num,
                                               self.image_format)
        if self.inference_status is True:
            x_test = get_dataset_summary(self.test_dataset_dir,
                                         image_format=self.image_format,
                                         inference_status=self.inference_status,
                                         sort=True)
            test_dataset = create_tf_dataset(x=x_test,
                                             y=None,
                                             batch_size=self.batch_size,
                                             img_height=self.image_height,
                                             img_width=self.image_width,
                                             img_channels=self.image_channels,
                                             encoder=self.char_to_num,
                                             image_format=self.image_format,
                                             inference_status=True)
            return train_dataset, validation_dataset, test_dataset
        else:
            return train_dataset, validation_dataset,

    def _create_model(self, inference_status, gpus):
        """Create model.

        Parameters:
            inference_status: bool,
            if true the model will be trained,
             else the model will be trained and inferred.
            gpus: int, Number of GPU used.

        Returns:
            train model or train and inference model
        """
        if gpus >= 2:
            strategy = MirroredStrategy()
            with strategy.scope():
                train_model = CRNN(self.image_width,
                                   self.image_height,
                                   self.image_channels,
                                   True,
                                   self.cnn_backbone,
                                   self.rnn_backbone)
                # Compile model
                train_model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        else:
            train_model = CRNN(self.image_width,
                               self.image_height,
                               self.image_channels,
                               len(self.characters),
                               True,
                               self.cnn_backbone,
                               self.rnn_backbone)
            # Compile model
            train_model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        if inference_status is True:
            inference_model = CRNN(self.image_width,
                                   self.image_height,
                                   self.image_channels,
                                   len(self.characters),
                                   False,
                                   self.cnn_backbone,
                                   self.rnn_backbone)
            return train_model, inference_model
        else:
            return train_model
