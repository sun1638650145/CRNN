# CRNN

An AutoML about CRNN, you can use three or four lines of code to recognize the captcha.

Read this in other languages：[简体中文](https://github.com/sun1638650145/CRNN/blob/master/README.md)、[English](https://github.com/sun1638650145/CRNN/blob/master/README-en.md)

## example

### pipeline

This is a very simple example.(Suitable for beginners)

```python
from CRNN import CRNNPipeline
pipeline = CRNNPipeline('./captcha_images_v2/')
pipeline.run()
```

The dataset used in the example, click [here](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip)

### custom

If you want higher accuracy and higher efficiency, you can use custom mode. (Suitable for experts)

1. You can use the model API and tools API to build your own model.
2. If you have any questions, welcome to communicate with the author and contact information qq:1638650145, email:s1638650145@gmail.com, and issue.

## Performance & optimization

1. For the previous version, the performance of Google’s caatcha image verification code has been improved by about 8.749 times, and the accuracy rate is about 60%.
2. Can use multiple GPUs.
3. The label length is inconsistent in a batch, so there is no need to explicitly define the label length.
4. Use tf.data.Datasets instead of yield to read in the datasets, theoretically it can handle data above 100G.
5. Separate the backbone of CNN and RNN and add more advanced backbone.

## If you want to do something

1. If you want to code, please use the PEP8, otherwise it must not pass.
2. If you want to use CPTN for post-detection identification, you should also communicate with the author. The contact information is above.
3. If you want to get the previous version of the code (it's too badly written, maybe you don't want it, anyway, there is no on GitHub), or communicate with the author, the contact information is also above.
4. If you want to star and fork, please don’t be stingy, your star’s fork is my driving force, thanks.