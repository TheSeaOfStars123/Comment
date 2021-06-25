# @Time : 2021/6/25 11:01 AM 
# @Author : zyc
# @File : main.py 
# @Title :
# @Description :

from config import opt
import torch
import matplotlib.pyplot as plt
from models import TextCNN
from data.dataset import CommentDataset
from torch.utils.data import TensorDataset,Dataset, DataLoader
from torch.autograd import Variable
from torchnet import meter
from data.preprocess import load_word2id, load_corpus_word2vec, load_corpus
from tqdm.notebook import tqdm
from sklearn import metrics
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(**kwargs, ):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    # step1 加载word2ix
    word2id = load_word2id(opt.word_to_id_path)
    # step2 加载word2vec
    word2vec = load_corpus_word2vec(opt.corpus_word2vec_path)
    # step3 加载train/val/test语料库
    # 可以比较一下两种加载数据的方式的不同
    train_contents, train_labels = load_corpus(opt.train_data_root, word2id, opt.max_len)  # train_contents:(19998, 50)
    train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
                                  torch.from_numpy(train_labels).type(torch.long))  # TensorDataset;tensors=tuple(tensor(19998,50),tensor(19998))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    print('train_dataset:', len(train_dataset))
    train_data = CommentDataset(opt.train_data_root, word2id, opt.max_len) # CommentDataset:contents/labels
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers)
    print('train_data:', len(train_data))
    validation_data = CommentDataset(opt.validation_data_root, word2id, opt.max_len)
    validation_loader = DataLoader(validation_data, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.num_workers)
    test_data = CommentDataset(opt.test_data_root, word2id, opt.max_len)
    # step4 模型
    model = TextCNN(word2vec)
    print(model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        print('Cuda is available!')
        model.cuda()
    # step5 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)
    # step6 统计指标
    best_accuracy = 0
    # step7 训练前准备工作
    # step8 训练
    for epoch in range(opt.max_epoch):
        train_loader = tqdm(train_loader)
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0], data[1]  # input:[16,50] labels:[16]
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        eval_outputs, eval_targets = eval_epoch(validation_loader, model)
        accuracy = metrics.accuracy_score(eval_targets, eval_outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            model.save()
            best_accuracy = accuracy

    # 可以计算下两种计算准确率方式的结果


def eval_epoch(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data[0], data[1]  # input:[16,50] labels:[16]
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            fin_targets.extend(labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

if __name__ == '__main__':
    train()