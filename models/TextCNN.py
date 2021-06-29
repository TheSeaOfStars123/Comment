# @Time : 2021/6/25 11:00 AM 
# @Author : zyc
# @File : TextCNN.py 
# @Title :
# @Description :

from .BasicModule import BasicModule
import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(BasicModule):
    def __init__(self, word2vec):
        super(TextCNN, self).__init__()

        # 使用预训练的词向量
        self.embedding = nn.Embedding(word2vec.shape[0], word2vec.shape[1])  # shape[0]是词表大小 shape[1]是50
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))  # 载入预训练模型的参数
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embedding.weight.requires_grad = True
        # 卷积层
        self.conv = nn.Conv2d(1, 256, (3, 50))  # 通道数/输出的深度/filter的高height/filter的宽width
        # Dropout
        self.dropout = nn.Dropout(0.2)
        # 全连接层
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        print('step1:x-dim',x.shape)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        y = self.conv(x)
        x = F.relu(self.conv(x)).squeeze(3)
        print('step3:x-size',x.size(2))
        z = F.max_pool1d(x, x.size(2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x