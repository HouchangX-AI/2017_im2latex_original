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
# 数据集预处理
在训练之前，需对formula进行分割。分割后的formula在Dataset中。含有.norm的均为分割后的formula.

# 实验结果
原文给出的实验结果如下，需要大量的数据样本作为支撑。可以看到，在training set size为10k时，test accuracy为25%左右。  
<img width="408" alt="test_accuracy" src="https://user-images.githubusercontent.com/37775638/77712783-c7f89980-700f-11ea-9489-386f6817b2ad.png">

本文采用的training set size为1000，vocab_size为278，实验结果如下
<img width="842" alt="test_predict" src="https://user-images.githubusercontent.com/37775638/77713387-3ab64480-7011-11ea-9b61-94c7e68f8b72.png">

bleu = 0.034904671179405385  
edit_distance = 0.18055868438837575

可以看出，由于实验所用training set size过小，导致训练结果比原文差。
