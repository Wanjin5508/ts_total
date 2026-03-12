>  基础知识在这个文档中整理，论文的核心内容参见[[论文特征清单]]的链接

# Types of time series classification methods
[[A Time Series Forest for Classification and Feature Extraction .pdf]]
## 1. instance-based
Instance-based classifiers predict a testing instance based on its *similarity* to the training instances. Among instance-based classifiers, *nearest-neighbor* classifiers with Euclidean distance (NNEuclidean) or **dynamic time warping** (NNDTW) have been widely and successfully used.

Usually NNDTW performs better than NNEuclidean (dynamic time warping [17] is robust to the distortion in the time axis), and is considered as a strong solution for time series problems [13]. Instancebased classifiers can be accurate, but they provide *limited insights into the temporal characteristics useful for classification.*


## 2. feature-based

Feature-based classifiers build models on temporal features, and potentially can be more *interpretable* than instance-based classifiers. Feature-based classifiers commonly consist of **two steps**: 
1. defining the temporal features and 
2. training a classifier based on the temporal features defined.

Nanopoulos et al. [11] extracted *statistical features* such as the mean and deviation of an entire time series, and then used a multi-layer perceptron neural network for classification. This method only captured the *global properties* of time series. *Local properties*, potentially informative for classification, were ignored.

# Time Series Forest
[[论文特征清单#1) TSF / Interval-based：最小但非常强的“区间手工特征”]]

TSF（Time Series Forest）里的“tree”指的就是**监督学习里的决策树（classification tree）**，并且 TSF 是一个**由很多棵决策树组成的集成**（forest），思路和 Random Forest 很像，但有两点关键区别：

1. **特征不是原始点**  
    TSF 每棵树不会直接用时间序列的每个采样点当特征，而是先随机抽取很多时间区间，然后在每个区间上计算少量统计特征（原论文是 3 个）：
    

- 区间均值（mean）
    
- 区间标准差（standard deviation）
    
- 区间斜率（slope，线性拟合斜率）  
    这些特征拼成该样本的特征向量，再喂给决策树。
    

2. **随机性主要来自“随机区间”**  
    Random Forest 的随机性更多来自“bootstrap + 随机选特征”。TSF 的核心随机性则来自：每棵树随机选一批区间（intervals），从而得到不同的特征子空间，再训练一棵决策树。
    

补充一点：原始 TSF 通常用的是类似 C4.5/CART 这种标准的分类树训练方式（例如用信息增益/基尼指数做划分），所以你完全可以把它理解成“**区间特征 + 决策树集成**”。

# Shapelets
Ye and Keogh [23] proposed time series shapelets to perform interpretable time series classification. *Shapelets* are time series **subsequences** which are in some sense maximally *representative* of a class [23]. Ye and Keogh [23], Xing et al. [22], Lines et al. [10] have successfully shown that time series shapelets can produce highly interpretable results.







