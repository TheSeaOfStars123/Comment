# @Time : 2021/6/25 10:59 AM 
# @Author : zyc
# @File : dataset.py 
# @Title :
# @Description :

from torch.utils import data
import numpy as np
import torch
from collections import Counter


class CommentDataset(data.Dataset):
    def __init__(self, path, word2id, max_len=50, classes=['0', '1']):
        self.path = path
        self.max_len = max_len
        self.word2id = word2id
        self.contents, self.labels = self.load_corpus()

    def __getitem__(self, idx: int):
        return self.contents[idx], self.labels[idx]

    def __len__(self):
        return len(self.contents)

    def load_corpus(self, classes=['0', '1']):
        """

        :param path: 样本语料库的文件
        :param word2ix: 语料文本中包含的词汇集
        :param max_len:
        :param classes:
        :return: 文本内容contents,以及分类标签labels(onehot形式)
        """
        contents, labels = [], []
        with open(self.path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                label = sp[0]  # train.txt/validation.txt的每行第一个字符是1/0
                content = [self.word2id[w] for w in sp[1:]]
                content = content[:self.max_len]
                if len(content) < self.max_len:
                    content += [self.word2id['_PAD_']] * (self.max_len - len(content))
                labels.append(label)
                contents.append(content)
        counter = Counter(labels)
        print('总样本数为：%d' % (len(labels)))
        print('各个类别样本数如下：')
        for w in counter:
            print(w, counter[w])

        contents = np.asarray(contents)
        cat2id = {cat: idx for (idx, cat) in enumerate(classes)}  # 0:'0' 1:'1'
        labels = np.asarray([cat2id[l] for l in labels])
        return contents, labels