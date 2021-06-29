# @Time : 2021/6/25 11:01 AM 
# @Author : zyc
# @File : main.py 
# @Title :
# @Description :

from config import opt
import torch
import jieba  # 分词
from models import TextCNN
from data.dataset import CommentDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from torch.utils.tensorboard import SummaryWriter
from utils.draw import draw_process
from data.preprocess import load_word2id, load_corpus_word2vec, load_corpus
from tqdm.notebook import tqdm
from sklearn import metrics
import time

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(**kwargs, ):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    # step1 加载word2ix
    word2id = load_word2id(opt.word_to_id_path)
    # step2 加载word2vec
    word2vec = load_corpus_word2vec(opt.corpus_word2vec_path)
    # step3 加载train/val语料库
    train_data = CommentDataset(opt.train_data_root, word2id, opt.max_len) # CommentDataset:contents/labels
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers)
    print('train_data:', len(train_data))
    validation_data = CommentDataset(opt.validation_data_root, word2id, opt.max_len)
    validation_loader = DataLoader(validation_data, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.num_workers)
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
    best_accuracy = 0  # 只保存准确率最高的模型
    loss_meter = meter.AverageValueMeter()
    top1 = meter.AverageValueMeter()

    # step7 训练前准备工作
    # 绘制图像需要
    training_loss = []
    train_accs = []
    valid_accs = []
    """
        保存训练日志
        路径:/tmp/training_log_0523_23:57:29.txt
        """
    prefix = opt.training_log + "_"
    path = time.strftime(prefix + '%m%d_%H:%M:%S.txt')
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.isfile(path):
        open(path, 'w')

    # step8 训练
    for epoch in range(opt.max_epoch):
        f = open(path, 'a')
        start = time.time()
        loss_meter.reset()
        train_loader = tqdm(train_loader)
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = Variable(data[0]), Variable(data[1])  # input:[16,50] labels:[16]
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            prec1 = metrics.accuracy_score(labels, outputs)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            top1.add(prec1.item())
            loss_meter.add((loss.item()))
            # 用于plt画图
            training_loss.append(loss_meter.value()[0])  # 记录每一个batch_size训练过程中的train_loss
            train_accs.append(top1.value()[0])

            # 使用tensorboard进行曲线绘制
            if not os.path.exists(opt.tensorboard_path):
                os.mkdir(opt.tensorboard_path)
            count = count + 1
            writer = SummaryWriter(opt.tensorboard_path)
            writer.add_scalar('Train/Loss', loss_meter.value()[0], count)
            writer.add_scalar('Train/Accuracy', top1.value()[0], count)
            writer.flush()

            if i % opt.print_freq == opt.print_freq - 1:
                print('[%d,%5d] train_loss :%.3f' %
                      (epoch + 1, i + 1, loss_meter.value()[0]))
                f.write('\n[%d,%5d] train_loss :%.3f' %
                        (epoch + 1, i + 1, loss_meter.value()[0]))

        # 当一个epoch结束之后开始打印信息
        # 计算验证集上的准确率
        eval_outputs, eval_targets = eval_epoch(validation_loader, model)
        accuracy = metrics.accuracy_score(eval_targets, eval_outputs)
        valid_accs.append(accuracy)
        print('epoch %d, lr %.4f, train_loss %.4f, train_acc %.3f %%, valid_acc %.3f %%, time %.1f sec' %
              (epoch + 1, lr, loss_meter.value()[0], top1.value()[0], accuracy, time.time() - start))
        f.write('\nepoch %d, lr %.4f, train_loss %.4f, train_acc %.3f %%, valid_acc %.3f %%, time %.1f sec' %
              (epoch + 1, lr, loss_meter.value()[0], top1.value()[0], accuracy, time.time() - start))
        f.close()

        # 保存当前最优模型
        if accuracy > best_accuracy:
            model.save()
            best_accuracy = accuracy

    # 所有迭代结束后使用plt画图像
    # 开始画图
    loss_iters = range(len(training_loss))
    train_acc_iters = range(len(train_accs))
    valid_acc_iters = range(len(valid_accs))
    draw_process('Comment_loss', loss_iters, training_loss, 'training_loss', opt.loss_file)
    draw_process('Commnet_acc', train_acc_iters, train_accs, 'training_acc', opt.train_acc_file)
    draw_process('Commnet_acc', valid_acc_iters, valid_accs, 'validation_acc', opt.valid_acc_file)


def eval_epoch(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for bi, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data[0], data[1]  # input:[16,50] labels:[16]
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            fin_targets.extend(labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist())
            # 评估过程中的准确率
            running_accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
            print('confuse_acc: %.6f' %(running_accuracy))
    return fin_outputs, fin_targets


def test(**kwargs, ):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    # step1 加载word2ix
    word2id = load_word2id(opt.word_to_id_path)
    # step2 加载word2vec
    word2vec = load_corpus_word2vec(opt.corpus_word2vec_path)
    # step3 加载test语料库
    test_data = CommentDataset(opt.test_data_root, word2id, opt.max_len)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_workers)
    # step4 模型
    model = TextCNN(word2vec)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path, map_location=torch.device('cpu')))
    if opt.use_gpu:
        model.cuda()
    model.eval()
    # step5 模型评估
    eval_outputs, eval_targets = eval_epoch(test_loader, model)
    # 评估指标：准确率 精确率 灵敏度(查全率/召回率) F1_score
    accuracy = metrics.accuracy_score(eval_targets, eval_outputs)
    precision = metrics.precision_score(eval_targets, eval_outputs)
    recall = metrics.recall_score(eval_targets, eval_outputs)
    F1 = metrics.f1_score(eval_targets, eval_outputs)
    print(f"Accuracy Score = {accuracy}, Precision Score = {precision}, Recell Score = {recall}, F1 score = {F1}")


# 使用模型对某条评价进行分类预测
def predict(comment_str):
    # step1 加载word2ix
    word2id = load_word2id(opt.word_to_id_path)
    # step2 加载word2vec
    word2vec = load_corpus_word2vec(opt.corpus_word2vec_path)
    # step3 加载test语料库
    test_data = CommentDataset(opt.test_data_root, word2id, opt.max_len)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.num_workers)
    # step4 模型
    model = TextCNN(word2vec)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path, map_location=torch.device('cpu')))
    if opt.use_gpu:
        model.cuda()
    # 将评论进行分词
    seg_list = jieba.lcut(comment_str, cut_all=False)
    words_to_idx = []
    for w in seg_list:
        try:
            index = word2id[w]
        except:
            index = 0
        words_to_idx.append(index)
    words_to_idx = words_to_idx[:opt.max_len]
    if len(words_to_idx) < opt.max_len:
        words_to_idx += [word2id['_PAD_']] * (opt.max_len - len(words_to_idx))
    inputs = torch.tensor(words_to_idx).reshape(1,-1)
    if opt.use_gpu:
        inputs = inputs.cuda()
    outputs = model(inputs)
    pred = outputs.argmax(axis=1).item()
    if(pred):
        print('Negative')
    else:
        print('Positive')
    return pred


if __name__ == '__main__':
    # 因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
    import shutil
    if os.path.exists(opt.tensorboard_path):
        shutil.rmtree(opt.tensorboard_path)
        os.mkdir(opt.tensorboard_path)
    # train()
    test()
    # comment_str1 = "《我不是药神》值得完整五星，叙事，表演，节奏，镜头都是年度最佳，唯一的短板可能就是树了正版药厂这个假敌人，但是国产电影嘛，也要理解，不算主动做恶，只能算被动装傻。"
    # comment_str2 = "比较反感这类戴着批判现实主义帽子的电影却使用了大量讨好大众的商业片常用媚俗技巧（重点表现在笑点与哭点的有意设计）"
    # predict(comment_str2)