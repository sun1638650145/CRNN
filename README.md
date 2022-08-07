# CRNN

An AutoML for CRNN, you can perform captcha recognition with just a few lines of code, read this in other languages: [简体中文](https://github.com/sun1638650145/CRNN/blob/master/README-zh.md)

## pipeline

This is a very simple example(Suitable for beginners).

```python
from CRNN import CRNNPipeline
pipeline = CRNNPipeline('./captcha_images_v2/')
pipeline.run()
```

The dataset used in the example, click [here](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip).

## custom

If you want higher accuracy and higher efficiency, you can use custom mode(Suitable for experts).

1. You can use the `model API` and `tools API` to build your own model, (View the source-code from reference ).
2. You are, welcome to communicate with the author.
