# @Time : 2021/6/25 11:00 AM 
# @Author : zyc
# @File : config.py 
# @Title :
# @Description :

import warnings
import os

class DefaultConfig(object):
    print("DefaultConfig:", os.getcwd())
    model = 'TextCNN'
    train_data_root = os.getcwd()+'/DataSet/train.txt'
    validation_data_root = os.getcwd()+'/DataSet/validation.txt'
    test_data_root = os.getcwd()+'/DataSet/test.txt'
    # wiki_word2vec_50_bin_path = '/Users/zyc/Desktop/情感分类/Dataset/wiki_word2vec_50.bin'
    wiki_word2vec_50_bin_path = os.getcwd()+'/DataSet/wiki_word2vec_50.bin'
    word_to_id_path = os.getcwd() + '/DataSet/word_to_id.txt'
    corpus_word2vec_path = os.getcwd() + '/DataSet/corpus_word2vec.txt'
    load_model_path = None

    raining_log = os.getcwd() + '/result/training_log'
    tensorboard_path = os.getcwd() + '/tmp/tensorboard'
    loss_file = os.getcwd() + '/result/result_loss'

    batch_size = 32
    use_gpu = True
    num_workers = 1
    print_frep = 500
    max_epoch = 4
    lr = 1e-3
    weight_decay = 1e-4  # 损失函数
    max_len = 50

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()