# 2017_im2latex_original
arXiv: 1609.04938

这个代码是根据arXiv: 1609.04938进行改编的，去除attention后的模型。encoder为multi-CNN，decoder为RNN。  
_att为原文中加了attention的模型。

# 训练和参数 
具体参数见params.py。

```
python main.py --train
python main_att.py --train
```
