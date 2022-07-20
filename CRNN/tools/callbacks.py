import os
from typing import List, Union

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


def get_callback(checkpoints_dir: Union[str, os.PathLike],
                 use_tensorboard: bool) -> List[Callback]:
    """Get the callback tool.

    Args:
        checkpoints_dir: str os.PathLike,
            the location where checkpoints are saved.
        use_tensorboard: bool,
            you can view some useful information on the TensorBoard.

    Return:
        List of keras.callbacks.Callback.
    """
    CALLBACKS = []

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    CALLBACKS.append(ModelCheckpoint(filepath=os.path.join(checkpoints_dir, 'model.{epoch:03d}_{val_loss:.2f}.h5'),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_freq='epoch'))

    if use_tensorboard:
        if not os.path.exists('./results/logs/'):
            os.makedirs('./results/logs/')
        CALLBACKS.append(TensorBoard(log_dir='./results/logs/',
                                     histogram_freq=1,
                                     write_graph=True,
                                     update_freq='epoch'))

    return CALLBACKS
