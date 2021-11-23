"""_coding:utf-8_"""
"""
Part one: Classic machine algorithms for recommender systems
Chapter one: Collaborative filtering algorithm (User-CF and Item-CF)
"""
__author__ = 'Junjie Wang'

import pandas as pd

def load_Data(path):
    '''
    加载数据
    input: document path
    output: user-item score matrix
    '''
    Data = pd.read_csv(path)
    Ratings = Data.pivot_table(index='userId', columns='movieId', values='rating')
    return Ratings

def pearson_similarity(UserItemMatrix, based='user'):
    '''
    计算皮尔逊系数
    param: 用户项目评分矩阵
    return：相似度矩阵
    '''
    if based == 'user':
        similarity = UserItemMatrix.T.corr(method='pearson')
    else:
        similarity = UserItemMatrix.corr(method='pearson')
    return similarity

def select_similar_user(UserSimilarity, k, uid):
    """
    基于用户的推荐电影
    :param UserSimilarity: 用户相似度
    :param k: k个相似度从高到低的用户
    :param uid: 基于uid的用户
    :return: 返回k个相似度从高到低的用户
    """
    UserSimilarity = UserSimilarity.loc[uid].dropna()
    SimilarUser = []
    for Uid in UserSimilarity.index:
        if UserSimilarity.loc[Uid] != None:
            SimilarUser.append([Uid, UserSimilarity.loc[Uid]])
    SimilarUser.sort(key=lambda x: -x[1])
    SimilarUser = [SimilarUser[i][0] for i in range(len(SimilarUser))]
    return SimilarUser[:k]

def user_likemovie(UserItemMatrix, uid):
    """
    :param UserItemMatrix: 用户评分矩阵
    :param uid: 用户id
    :return: 用户最喜欢的电影降序排序
    """
    like_movies = []
    movies = UserItemMatrix.loc[uid].dropna()
    for index in movies.index:
        like_movies.append([index, movies.loc[index]])
    like_movies.sort(key=lambda x: -x[1])
    like_movies = [like_movies[i][0] for i in range(len(like_movies))]
    return list(like_movies)

def recommend_movie(uid, UserSimilarity, k, UserItemMatrix):
    """
    :param uid: 用户id
    :param UserSimilarity: 用户相似性矩阵
    :param k: 选出兴趣最相似的k个用户
    :param UserItemMatrix: 用户评分矩阵
    :return: 给uid用户推荐其未看过的电影id，该电影得是看过的超过1/5相似用户个数的电影
    """
    similar_users = select_similar_user(UserSimilarity, k, uid)
    moviesHold = []
    for similar_user in similar_users:
        moviesHold.append(user_likemovie(UserItemMatrix, similar_user))
    selfs_movies = list(UserItemMatrix.loc[uid].dropna().index)
    movies_count = {}
    recommend_movies = []
    for movies in moviesHold:
        for movie in movies:
            if movie not in movies_count:
                movies_count[movie] = 1
            else:
                movies_count[movie] += 1
    for movie in movies_count:
        if movies_count[movie] > k // 5:
            recommend_movies.append(movie)
    return list(set(recommend_movies) & set(selfs_movies))

def predict_score(uid, iid, UserItemMatrix, similarity):
    """
    预测给定用户对给定物品的评分: 基于用户相似度
    """
    similar_users = similarity[uid].drop([uid]).dropna()
    similar_users = similar_users.where(similar_users > 0.30).dropna()
    indexes = set(UserItemMatrix[iid].dropna().index) & set(similar_users.index)  # 筛选出看过该部电影的用户
    similar_users = similar_users.loc[list(indexes)]

    sum_up = 0
    sum_down = 0
    for sim_uid, similarity in similar_users.iteritems():
        sim_user_similarity_movie = UserItemMatrix.loc[sim_uid]
        sim_user_rating_for_item = sim_user_similarity_movie[iid]
        sum_up += similarity * sim_user_rating_for_item
        sum_down += similarity
    return sum_up / (sum_down + 1e-3)

def predict_all(UserItemMatrix, similarity):
    '''
    预测全部用户对全部物品的评分
    return: 评分矩阵，根据已知的评分计算评分的准确度
    '''
    user_id = UserItemMatrix.index
    item_id = UserItemMatrix.columns
    Predict_Matrix = pd.DataFrame(index=user_id, columns=item_id)
    error = 0
    actual_value = 0
    for uid in user_id:
        for iid in item_id:
            Predict_Matrix.loc[uid].loc[iid] = predict_score(uid, iid, UserItemMatrix, similarity)
            if UserItemMatrix.loc[uid].loc[iid] != None:
                error += abs(UserItemMatrix.loc[uid].loc[iid] - Predict_Matrix.loc[uid].loc[iid])
                actual_value += UserItemMatrix.loc[uid].loc[iid]
    accuracy = 1 - error / actual_value
    return Predict_Matrix, accuracy

if __name__ == '__main__':
    UserItemMatrix = load_Data('ratings.csv')
    UserSimilarity = pearson_similarity(UserItemMatrix, based='user')
    movies = recommend_movie(1, UserSimilarity, 20, UserItemMatrix)
    print(movies)