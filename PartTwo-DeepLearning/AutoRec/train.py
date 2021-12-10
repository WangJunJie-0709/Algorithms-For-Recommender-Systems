import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import dataset, dataloader
import numpy as np
from DataProcess import *
from model import AutoRec

def train(model, Train_loader, Val_loader, optimizer, loss_fn, lr, batchsize, epoches=100):
    model.train()
    for epoch in range(epoches):
        print("第{}轮训练开始，共{}轮".format(epoch + 1, epoches))
        train_loss = 0
        for i, sample in enumerate(Train_loader):
            x = Variable(sample['x'].type(torch.FloatTensor))
            y = Variable(sample['y'])
            output = model(x)
            loss = loss_fn(output, y.float())

            # 优化器优化模型
            optimal.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 开始验证模式
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for i, sample in enumerate(Val_loader):
                x_val = Variable(sample['x'].type(torch.FloatTensor))
                y_val = Variable(sample['y'])

                output = model(x_val)
                loss = loss_fn(output, y_val.float())
                eval_loss += loss
        print('第{}轮训练完成， 训练损失为{}， 验证损失为{}'.format(epoch + 1, train_loss, eval_loss))
    return model


if __name__ == '__main__':
    UserItemMatrix = load_data(r'G:\github项目\Recomendation_system\Data\movie\ratings.csv')
    ItemSimilarity = person_similarity(UserItemMatrix, based='item')
    similar_item = select_similar_item(ItemSimilarity, 10, 3)
    need_user = looked_user(UserItemMatrix, 1)
    pre_score = predict_score(1, 1, UserItemMatrix, ItemSimilarity)  # 预测uid为1对iid为1的评分，实际为4
    Pre, acc = predict_all(UserItemMatrix, ItemSimilarity)
    Pre = add_zeros(UserItemMatrix, Pre)
    Train_Data, Val_Data = split_data(Pre)

    train_set = MyDataset(Train_Data)
    val_set = MyDataset(Val_Data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 1  # 每一批的数据数量

    # 定义模型
    num_users = Train_Data.shape[0]
    num_items = Train_Data.shape[1]
    model = AutoRec(num_users, num_items, 300)
    # AutoRec = AutoRec.to(device)

    # 定义损失函数
    loss_fn = nn.MSELoss()
    # loss_fn = loss_fn.to(device)

    # 定义优化器
    lr = 0.01  # 学习率
    optimal = optim.SGD(model.parameters(), lr)

    # 创建数据迭代集
    Train_loader = DataLoader(train_set, batch_size=batchsize)
    Val_loader = DataLoader(val_set, batch_size=batchsize)

    train(model, Train_loader, Val_loader, optimal, loss_fn, lr, batchsize)