# CRNN

一个关于CRNN的AutoML, 你可以使用三四行代码就可以进行验证码的识别, 阅读其他语言版: [English](https://github.com/sun1638650145/CRNN/blob/master/README.md)

## 流水线

这是一个非常简短的例子(入门推荐).

```python
from CRNN import CRNNPipeline
pipeline = CRNNPipeline('./captcha_images_v2/')
pipeline.run()
```

例子中使用的数据集在这里[点击](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip).

## 自定义

如果你希望更高的准确率和更高的效率, 可以使用自定义模式(针对有经验的开发者).

1. 你可以使用`model API`和`tools API`构建你自己的模型.
2. 欢迎和作者联系/交流.
