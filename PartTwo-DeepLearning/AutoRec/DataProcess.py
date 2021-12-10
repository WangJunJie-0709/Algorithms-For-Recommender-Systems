import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

# 使用协同过滤算法生成训练样本, 使用ItemCF
def load_data(path):
    Data = pd.read_csv(path, nrows=1000)
    Ratings = Data.pivot_table(index='userId', columns='movieId', values='rating')
    return Ratings

def person_similarity(UserItemMatrix, based='user'):
    """计算皮尔逊相关系数，这里是基于用户的UserCF，若计算ItemCF可改为item"""
    if based == 'user':
        similarity = UserItemMatrix.T.corr(method='pearson') # corr是按列计算相关性，故这里计算用户相关性需要将用户矩阵转置
    else:
        similarity = UserItemMatrix.corr(method='pearson')
    return similarity


def select_similar_item(ItemSimilarity, k, iid):
    ItemSimilarity = ItemSimilarity.loc[iid].dropna()
    SimilarItem = []
    for IId in ItemSimilarity.index:
        if ItemSimilarity.loc[IId] != None:
            SimilarItem.append([IId, ItemSimilarity.loc[IId]])
    SimilarItem.sort(key=lambda x: -x[1])
    SimilarItem = [SimilarItem[i][0] for i in range(len(SimilarItem))]
    return SimilarItem[:k]


def looked_user(UserItemMatrix, iid):
    looked_users = []
    users = UserItemMatrix.loc[:,iid].dropna()
    for index in users.index:
        looked_users.append([index, users.loc[index]])
    looked_users.sort(key=lambda x: -x[1])
    looked_users = [looked_users[i][0] for i in range(len(looked_users))]
    return looked_users


def predict_score(uid, iid, UserItemMatrix, similarity):
    similar_item = similarity.loc[iid].drop([iid]).dropna()
    similar_item = similar_item.where(similar_item >= 0.1).dropna()
    indexes = set(UserItemMatrix.loc[uid].dropna().index) & set(similar_item.index) # 筛选出对相似物品有过评分的用户
    similar_item = similar_item.loc[list(indexes)]

    sum_up = 0
    sum_down = 0
    for sim_iid, similarity in similar_item.items():
        sim_item_similarity_user = UserItemMatrix.loc[:,sim_iid]
        sim_item_rating_for_user = sim_item_similarity_user.loc[uid]
        sum_up += similarity * sim_item_rating_for_user
        sum_down += similarity
    return sum_up / (sum_down + 1e-5)


def predict_all(UserItemMatrix, similarity):
    """预测全部用户对物品的评分"""
    user_id = UserItemMatrix.index
    item_id = UserItemMatrix.columns
    Predict_Matrix = pd.DataFrame(index=user_id, columns=item_id) # 新建一个与原先用户评分矩阵相同的空DataFrame
    error = 0
    actual_value = 0
    for uid in user_id:
        for iid in item_id:
            Predict_Matrix.loc[uid].loc[iid] = predict_score(uid, iid, UserItemMatrix, similarity)
            if UserItemMatrix.loc[uid].loc[iid] != None:
                error += abs(UserItemMatrix.loc[uid].loc[iid] - Predict_Matrix.loc[uid].loc[iid])  # 按照已有的评分进行准确率评估
                actual_value += UserItemMatrix.loc[uid].loc[iid]
    accuracy = error / actual_value
    return Predict_Matrix, accuracy

def add_zeros(UserItemMatrix, Pre):
    # 那些0值用每一个物品的平均评分来代替
    ind = UserItemMatrix.index
    col = UserItemMatrix.columns
    average = []
    for iid in UserItemMatrix.columns:
        average.append(sum(Pre.loc[:,iid]) / len(Pre.loc[:,iid]))
    for uid in Pre.index:
        for i in range(len(Pre.columns)):
            if Pre.loc[uid].iloc[i] == 0:
                Pre.loc[uid].iloc[i] = average[i]
    return Pre


def split_data(Data, ratio=0.8):
    Train_Data = Data.loc[:,:len(Data.columns) * ratio].T
    Val_Data = Data.loc[:, len(Data.columns)*ratio:].T
    Train_Data = Train_Data.values
    Val_Data = Val_Data.values
    Train_Data = torch.from_numpy(Train_Data.astype(float))
    Val_Data = torch.from_numpy(Val_Data.astype(float))
    return Train_Data, Val_Data



class MyDataset(Dataset):
    def __init__(self, Data):
        self.Data = Data
        self.x = Data
        self.y = Data

    def __getitem__(self, index):
        sample = {'x': self.x[index], 'y': self.y[index]}
        return sample

    def __len__(self):
        return len(self.Data)


if __name__ == '__main__':
    UserItemMatrix = load_data(r'G:\github项目\Recomendation_system\Data\movie\ratings.csv')
    ItemSimilarity = person_similarity(UserItemMatrix, based='item')
    similar_item = select_similar_item(ItemSimilarity, 10, 3)
    need_user = looked_user(UserItemMatrix, 1)
    pre_score = predict_score(1, 1, UserItemMatrix, ItemSimilarity)  # 预测uid为1对iid为1的评分，实际为4
    Pre, acc = predict_all(UserItemMatrix, ItemSimilarity)
    Pre = add_zeros(UserItemMatrix, Pre)
    Train_Data, Val_Data = split_data(Pre)