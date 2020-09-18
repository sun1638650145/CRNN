# CRNN

一个关于CRNN的AutoML，你可以使用三四行代码就可以进行验证码的识别

## 例子

### pipeline

这是一个非常简短的例子（入门推荐）

```python
from CRNN import CRNNPipeline
pipeline = CRNNPipeline('./captcha_images_v2/')
pipeline.run()
```

例子中使用的数据集在点击[这里](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip)

### custom

如果你希望更高的准确率和更高的效率，可以使用自定义模式（针对有经验的开发者）

1. 你可以使用model和tools下的API构建你自己的模型
2. 有问题，欢迎和作者交流，联系方式qq:1638650145，邮箱:s1638650145@gmail.com

## 性能和优化

1. 针对上一个版本在谷歌的capatcha上性能提高8.749倍左右
2. 可以使用多GPU的并行解决方案
3. 在一个batch内标签长度不一致，不用显式的定义标签的长度
4. 使用tf.data.Datasets代替yield读入数据集，理论上可以处理100G以上的数据
5. 分离CNN和RNN的backbone，增加更先进的backbone

## 如果你想

1. 如果你想改进代码，请使用PEP8标准，否则一定无法通过
2. 如果你想使用CPTN进行检测后识别，也要与作者交流，联系方式在上面
3. 如果你想获得上一版本的代码（写得太烂估计你也不想要，反正GitHub上没有），还是与作者交流，联系方式在上面
4. 如果你想star和fork，请不要吝惜，你的star的fork就是我前进的动力，谢谢