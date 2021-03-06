# 一、GBDT模型

集成学习是一种协同多个学习器完成任务的学习方法，其原理是使用某一种方式将多个学习器进行集成，以此获得比单一学习器更优的泛化性能。梯度提升决策树(GBDT)是一种Boost集成学习算法，其核心思想是通过多轮迭代产生多个弱分类器，在每一次迭代后计算损失函数的负梯度，将其作为残差的近似值。在GBDT分类模型中，一般使用CART回归树作为基学习器，每个分类学习器的训练都是基于上一轮分类器预测结果的残差，以串行的方式向残差减小的方向进行梯度迭代，最后将每个弱分类器得到的结果进行加权求和得到最终的分类器。

GBDT是采用加法模型（基函数的线性组合），以及不断减小训练误差来达到回归或者分类的算法，其训练过程如下：
    
![image](https://user-images.githubusercontent.com/93982957/146362271-078b8330-133b-4ee6-9bd3-1d301ae27031.png)

# 1、构建分类GBDT步骤：

(1)初始化GBDT

和回归问题一样，分类GBDT的初始状态仅有一个叶子结点，该节点为所有样本的初始预测值

![image](https://user-images.githubusercontent.com/93982957/146362817-d8b5958d-2761-4b93-b50d-5bc7b2aa970c.png)

(2)循环生成决策树

    1)计算负梯度得到残差
    
    2)使用CART回归树来拟合残差
    
    3)对于每一个叶子结点，计算最佳残差拟合值
    
    4)更新模型
    
# 二、LR模型

逻辑回归模型(LR)是一种基于回归分析的分类算法，LR算法使用sigmoid激活函数将线性回归的分析结果转换为概率值。逻辑回归假设数据服从伯努利分布，通过极大化似然估计函数的方法，使用梯度下降方法来求解参数，来达到将数据二分类的目的

相比于协同过滤和矩阵分解利用用户和物品的相似度进行推荐，逻辑回归模型将问题看成了一个分类问题，通过预测正样本的概率对物品进行排序。可以理解为用户点击或者不点击，即逻辑回归模型将用户推荐问题转换成了点击率预估问题

流程：

    (1)将用户的年龄、性别、使用时间、使用地点等离散型特征转换为数值型特征
    
    (2)确定逻辑回归的优化目标，比如将点击率转换为二分类问题，这样就可以得到分类问题常用的损失函数，如交叉熵损失函数，进而训练模型
    
    (3)在预测的时候，将特征向量输入模型产生预测结果，得到用户点击物品的概率
    
    (4)利用点击概率对候选物品进行排序，得到推荐列表
    
![image](https://user-images.githubusercontent.com/93982957/146364155-4107c9c2-b5fc-411d-982d-efae1b612e51.png)

优点：

    (1)LR模型形式简单，可解释性好
    
    (2)训练时便于并行化
    
    (3)资源占用小，比较省内存
    
    (4)方便输出结果调整
    
缺点

    (1)表达能力不强，无法进行特征交叉
    
    (2)准确率不高
    
    (3)处理非线性模型比较麻烦
    
    (4)需要进行大量特征工程
    
# 三、GBDT+LR模型

LR算法属于线性模型，模型简单，能够处理海量数据，但是仅在有良好线性关系的数据集上有效，其学习能力有限，需要进行大量特征工程，容易欠拟合。而GBDT可以自动产生有区分度的特征，避免复杂的人工成本。

GBDT+LR模型使用最广泛的场景是CTR点击率预估，即预测当给用户推荐某广告时会不会被点击

训练步骤：

    (1)利用原始数据集训练GBDT模型构造一系列的决策树，组成一个强分类器
    
    (2)利用训练好的GBDT模型对原始数据进行预测时，不以分类概率为输出，而是以模型中每棵树的预测值所属叶节点的位置为新特征提取特征值，生成新的数据集
    
    (3)对新数据进行one-hot编码，所有的样本的输出会组成一个标记每颗决策树输出的叶结点位置的稀疏矩阵
    
    (4)将该矩阵作为新的训练数据作为LR的训练集进行训练
    
