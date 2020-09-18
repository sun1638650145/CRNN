import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


def get_callback(checkpoint_dir, tensorboard):
    """Get callback list.

    Args:
        checkpoint_dir: str or None,
            the checkpoint will be saved in this path.
        tensorboard:
            You can view some useful information on the localhost.
    Returns:
        List of keras.callbacks.
    """
    CALLBACKS = []

    if tensorboard is True:
        if os.path.exists('./logs/') is False:
            os.mkdir('./logs/')
        CALLBACKS.append(
            TensorBoard(log_dir='./logs/',
                        histogram_freq=1,
                        write_graph=True,
                        update_freq='epoch')
        )

    if checkpoint_dir is not None:
        if checkpoint_dir[len(checkpoint_dir) - 1] is not '/':
            checkpoint_dir += '/'
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)
        CALLBACKS.append(
            ModelCheckpoint(filepath=checkpoint_dir+'model.{epoch:04d}-{val_loss:.04f}.h5',
                            monitor='val_loss',
                            verbose=1,
                            period=5),
        )

    return CALLBACKS