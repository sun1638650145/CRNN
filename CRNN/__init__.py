"""An AutoML for CRNN, you can perform captcha recognition
 with just a few lines of code."""
__version__ = '2.1'

import logging

# 配置logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(name='CRNN')

from CRNN.pipeline import CRNNPipeline
