# 图像描述生成
## 安装
spacy需要的语言模型`en-core-web-sm`可以用提供的whl直接安装`pip install en_core_web_sm-3.3.0-py3-none-any.whl`。

## 数据集
训练集: Kaggle [flickr8k](https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions) 数据集。  
测试集: Kaggle [flickr30k](https://www.kaggle.com/datasets/nunenuh/flickr30k)中抽取120张图片

## 训练
`python train.py`  
训练集图片和图片标注文件需要在train.py的25, 26行指定。默认为`flickr8k/images`,`flickr8k/captions.txt`。

## 测试
`bleu.ipynb`中放置了一些测试代码，测试图像数据需要存放在`./test/images`中，图像描述文件需要保存为`./test/captions.txt`

或

`python test.py`
使用gradio构建了一个简单的应用程序。

或

访问已部署好的在线demo: https://huggingface.co/spaces/leonhardt/ImageCaptionDemo