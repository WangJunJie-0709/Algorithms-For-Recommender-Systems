# 协同过滤

# 一、什么是协同过滤

协同过滤是协同大家的反馈、评价和意见一起对海量的信息进行过滤，从中筛选出目标用户可能感兴趣的信息的推荐过程。

# 协同过滤推荐过程：

    (1)用户访问某网站，网站的推荐系统需要决定是否推荐A物品给该用户。其中可以利用的数据有该用户对其他商品的历史评价数据，以及其他用户对这些商品的历史评价数据。
  
    (2)用共现矩阵表示用户-物品评分矩阵，用户作为矩阵行坐标，商品作为列坐标，用户对物品的行为数据作为矩阵中相应的元素值
  
    (3)生成共现矩阵后，推荐问题就转换成了预测矩阵中缺失值的问题。预测的第一步是找到与该用户兴趣最相似的n(Top n用户)个用户，然后综合相似用户对该物品的评价，得到该用户对待推荐物品的评价预测。
  
协同过滤是一个非常直观、可解释性很强的模型，但它并不具备较强的泛化性。这就导致了一个比较严重的问题：热门物品具有很强的头部效应，荣誉跟大量物品产生相似性，而尾部的物品由于特征向量稀疏，很少与其他物品产生相似性，导致很少被推荐。
  
# 二、用户相似度计算

1、余弦相似度：

![image](https://user-images.githubusercontent.com/93982957/146889607-02b04d68-5617-4b1e-87ca-9a1462dc0667.png)

余弦相似度衡量用户向量x和用户向量y之间的向量夹角大小。显然，夹角越小，证明余弦相似度越大，两个用户越相似

2、皮尔逊相关系数：

![image](https://user-images.githubusercontent.com/93982957/146890568-80539a67-4bb3-4613-8f1a-0fc9c1822b58.png)

相比余弦相似度，皮尔逊相关系数通过使用用户平均分对各独立评分进行修正，减小了用户评分偏置的影响。

# 三、最终结果的排序

在获得Top n相似用户之后，利用Top n相似用户生成最终推荐结果的过程如下：

假设"目标用户与其相似用户的喜好使相似的"，可根据相似用户的已有评价对目标用户的偏好进行预测。在获得用户对不同物品的评价预测之后，最终的推荐列表根据预测得分进行排序即可得到。

# 用户协同过滤的缺点
  
    1、在互联网应用的场景下，用户数往往远大于物品数，而UserCF需要维护用户相似度矩阵以便快速找出Topn相似用户。该用户相似度矩阵的存储开销非常大。
    
    2、用户的历史数据向量往往非常稀疏，对于只有几次购买或者点击行为的用户来说，找到相似用户的准确度是非常低的，这导致UserCF不适用于那些正反馈获取困难的应用场景（例如酒店预订、大件商品购买等低频应用）
    
# 四、ItemCF

# ItemCF的具体步骤

    (1)基于历史数据，构建以用户（m个）为行坐标，物品（n个）为列坐标的m×n维的共现矩阵  
    
    (2)计算共现矩阵两两列向量的相似性，构建n×n维的物品相似度矩阵
    
    (3)获得用户历史行为数据中的正反馈物品列表
    
    (4)利用物品相似度矩阵，针对目标用户历史行为中的正反馈物品，找出相似的TopK个物品，组成相似物品集合
    
    (5)对相似物品集合中的物品，利用相似度分值进行排序，生成最终的推荐列表
    
# 五、UserCF与ItemCF的应用场景

一方面，由于UserCF基于用户相似度进行推荐，使其具备更强的社交特性，用户能够快速地得知与自己兴趣相似的人最近喜欢什么。这样的特点非常适用于新闻推荐场景。因为新闻本身的兴趣点往往是分散的，相比用户对不同新闻的兴趣偏好，新闻的及时性、热点性往往是其最重要的属性，而UserCF正适用于发现热点，以及跟踪热点的趋势。

另一方面， ItemCF更适用于兴趣变化比较稳定的应用。在Netflix的视频推荐场景中，用户观看电影、电视剧的兴趣点往往比较稳定，因此利用ItemCF推荐风格、类型相似的视频是更合理的选择。



    