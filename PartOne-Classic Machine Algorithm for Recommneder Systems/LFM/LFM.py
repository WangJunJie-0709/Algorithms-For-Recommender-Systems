"""_coding:utf-8_"""
"""
Part one: Classic machine algorithms for recommender systems
Chapter two: LFM and SVD
"""
__author__ = 'Junjie Wang'

import pandas as pd
import numpy as np
import random

def load_Data(path):
    """
    :param path: 文件路径
    :return: 返回原始的用户评分矩阵
    """
    Data = pd.read_csv(path, nrows=10000)
    Ratings = Data.pivot_table(index='userId', columns='movieId', values='rating')
    return Ratings

class LFM(object):

    def __init__(self, Rating_Data, F, learn_rate=0.1, lmbd=0.1, max_iteration=50):
        """
        隐语义模型，使用随机梯度下降法进行矩阵更新
        :param Rating_Data:
        :param F: 隐特征个数
        :param learn_rate: 学习率
        :param lmba: 正则化
        :param max_iteration: 最大迭代次数
        """
        self.F = F
        self.learn_rate = learn_rate
        self.lmbd = lmbd
        self.Rating_Data = Rating_Data
        self.max_iteration = max_iteration

        self.features = []
        for i in range(self.F):
            self.features.append("feature{}".format(i+1))
        users_id = Rating_Data.index
        items_id = Rating_Data.columns

        # 初始化随机用户和物品隐语义矩阵
        self.P = pd.DataFrame(np.random.rand(len(users_id), len(self.features)), index=users_id, columns=self.features)
        self.Q = pd.DataFrame(np.random.rand(len(self.features), len(items_id)), index=self.features, columns=items_id)

    def train(self):
        """
        随机梯度下降法训练用户和物品隐矩阵
        :return: 返回更新完毕的隐语义矩阵
        """
        for step in range(self.max_iteration):
            total_error = 0
            for user_id in self.Rating_Data.index:
                random_items_id = random.sample(list(self.Rating_Data.loc[user_id].dropna().index), 1)
                for random_item_id in random_items_id:
                    predict_point = self.predict(user_id, random_item_id)
                    true_point = self.Rating_Data.loc[user_id].loc[random_item_id]
                    user_error = true_point - predict_point
                    total_error += abs(user_error)
                    for f in range(self.F):
                        self.P.loc[user_id].iloc[f] += self.learn_rate * (user_error * self.Q.iloc[f].loc[random_item_id] -
                                                                                 self.lmbd * self.P.loc[user_id].iloc[f])
                        self.Q.iloc[f].loc[random_item_id] += self.learn_rate * (user_error * self.P.loc[user_id].iloc[f] -
                                                                          self.lmbd * self.Q.iloc[f].loc[random_item_id])
            for item_id in self.Rating_Data.columns:
                random_users_id = random.sample(list(self.Rating_Data.loc[:, item_id].dropna().index), 1)
                for random_user_id in random_users_id:
                    predict_point = self.predict(random_user_id, item_id)
                    true_point = self.Rating_Data.loc[random_user_id].loc[item_id]
                    item_error = true_point - predict_point
                    total_error += abs(item_error)
                    for f in range(self.F):
                        self.P.loc[random_user_id].iloc[f] += self.learn_rate * (item_error * self.Q.iloc[f].loc[item_id] -
                                                                         self.lmbd * self.P.loc[random_user_id].iloc[f])
                        self.Q.iloc[f].loc[item_id] += self.learn_rate * (item_error * self.P.loc[random_user_id].iloc[f] -
                                                                     self.lmbd * self.Q.iloc[f].loc[item_id])
            self.learn_rate *= 0.9
            print("{}次迭代----------error:{}".format(step+1, total_error))

    def predict(self, user_id, item_id):
        """
        :param user_id: 用户id
        :param item_id: 物品id
        :return: 返回用户对物品的预测评分
        """
        return sum(self.P.loc[user_id].iloc[f] * self.Q.iloc[f].loc[item_id] for f in range(self.F))

class SVD(object):
    """
    带偏置的LFM
    """
    def __init__(self, Rating_Data, F, learn_rate=0.0001, lmbd=0.00001, max_iteration=20):
        """
        隐语义模型，使用随机梯度下降法进行矩阵更新
        :param Rating_Data:
        :param F: 隐特征个数
        :param learn_rate: 学习率
        :param lmba: 正则化
        :param max_iteration: 最大迭代次数
        """
        self.F = F
        self.learn_rate = learn_rate
        self.lmbd = lmbd
        self.Rating_Data = Rating_Data
        self.max_iteration = max_iteration

        self.features = []
        for i in range(self.F):
            self.features.append("feature{}".format(i+1))
        users_id = Rating_Data.index
        items_id = Rating_Data.columns

        # 初始化随机用户和物品隐语义矩阵
        self.P = pd.DataFrame(np.random.rand(len(users_id), len(self.features)), index=users_id, columns=self.features)
        self.Q = pd.DataFrame(np.random.rand(len(self.features), len(items_id)), index=self.features, columns=items_id)

        # 初始化偏置
        self.bu = pd.DataFrame(np.random.rand(len(users_id), 1), index=users_id, columns=['userbias'])  # 用户偏置
        self.bi = pd.DataFrame(np.random.rand(1, len(items_id)), index=['itembias'], columns=items_id)  # 物品偏置

        # 得到全局平均值
        self.sum = 0
        count = 0
        for uid in users_id:
            for iid in items_id:
                if Rating_Data.loc[uid].loc[iid] != 0:
                    self.sum += Rating_Data.loc[uid].loc[iid]
                    count += 1
        self.u = round(self.sum / count, 6)

    def train(self):
        """
        随机梯度下降法训练用户和物品隐矩阵
        :return: 返回更新完毕的隐语义矩阵
        """
        for step in range(self.max_iteration):
            total_error = 0
            for user_id in self.Rating_Data.index:
                random_items_id = random.sample(list(self.Rating_Data.loc[user_id].dropna().index), 1)
                for random_item_id in random_items_id:
                    predict_point = self.predict(user_id, random_item_id)
                    true_point = self.Rating_Data.loc[user_id].loc[random_item_id]
                    user_error = true_point - predict_point
                    total_error += abs(user_error)
                    self.bu.loc[user_id] += self.learn_rate * (user_error - self.lmbd * self.bu.loc[user_id])
                    self.bi.loc[:, random_item_id] += self.learn_rate * (user_error - self.lmbd * self.bi.loc[:, random_item_id])
                    for f in range(self.F):
                        self.P.loc[user_id].iloc[f] += self.learn_rate * (user_error * self.Q.iloc[f].loc[random_item_id]
                                                                          - self.lmbd * self.P.loc[user_id].iloc[f])
                        self.Q.iloc[f].loc[random_item_id] += self.learn_rate * (user_error * self.P.loc[user_id].iloc[f]
                                                                                 - self.lmbd * self.Q.iloc[f].loc[random_item_id])

            # 对物品隐矩阵进行随机梯度下降更新
            for item_id in self.Rating_Data.columns:
                random_users_id = random.sample(list(self.Rating_Data.loc[:, item_id].dropna().index), 5)
                for random_user_id in random_users_id:
                    predict_point = self.predict(random_user_id, item_id)
                    true_point = self.Rating_Data.loc[random_user_id].loc[item_id]
                    item_error = true_point - predict_point
                    total_error += abs(item_error)
                    self.bu.loc[random_user_id] += self.learn_rate * (item_error - self.lmbd * self.bu.loc[random_user_id])
                    self.bi.loc[:, item_id] += self.learn_rate * (item_error - self.lmbd * self.bi.loc[:, item_id])
                    for f in range(self.F):
                        self.P.loc[random_user_id].iloc[f] += self.learn_rate * (item_error * self.Q.iloc[f].loc[item_id] -
                                                                         self.lmbd * self.P.loc[random_user_id].iloc[f])
                        self.Q.iloc[f].loc[item_id] += self.learn_rate * (item_error * self.P.loc[random_user_id].iloc[f] -
                                                                     self.lmbd * self.Q.iloc[f].loc[item_id])
            print("{}次迭代----------error:{}".format(step+1, total_error))

    def predict(self, user_id, item_id):
        """
        :param user_id: 用户id
        :param item_id: 物品id
        :return: 返回用户对物品的预测评分
        """
        return self.u + self.bu.loc[user_id].iloc[0] + self.bi.iloc[0].loc[item_id] + \
               sum(self.P.loc[user_id].iloc[f] * self.Q.iloc[f].loc[item_id] for f in range(self.F))

if __name__ == '__main__':
    path = r'G:\github项目\Recomendation_system\Data\movie\ratings.csv'
    Rating_Data = load_Data(path)  # 610 x 9724
    svd = SVD(Rating_Data, 5)
    svd.train()
    pre = svd.predict(1, 1)
    print(pre)