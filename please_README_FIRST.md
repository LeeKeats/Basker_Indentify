# 关于Baseline的修改

## 一.有关赛题的baseline的修改

赛题方已经对baseline进行了修改，但修改只是在code的process逻辑结构上，并没有对赛题的core进行改动，这里建议仍使用原赛题，保证项目的鲁棒性（这里指可进行模型融合的项目）

对于单模型优化项目，建议与新赛题对齐

## 二.有关对baseline的修改

1.有关.configs/BallShow

添加resnet50的配置文件：resnet50_ibn.yml,这里不需要调整参数，本来也不是主分支，所以没意义修改

transreid的配置文件没有进行任何修改！！！

2..model/backbone && .model.make_model

backbone：resnet_ibn.py 

make_model : 添加了resnet分支

3.utils/reranking ：由于距离级融合进行了必要的改进

4.增加了tools分支，内部是特征提取的脚本，需要可用

5.test.py

直接利用最优参数

```
#运行
(your_own_env)you_own_root>:python test.py
```

## 三.权重

log目录里有1个transreid权重，和两个resnet权重，".logs\BallShow_resnet50_ibn_v2"中的为最优权重

