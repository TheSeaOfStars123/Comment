# @Time : 2021/6/25 11:09 AM 
# @Author : zyc
# @File : preprocess.py 
# @Title :
# @Description : 数据预处理：构建词汇表并存储

from config import opt
import numpy as np
from collections import Counter


def build_word2id(save_file_path):
    """
    :param file: word2id保存地址
    :return:
    """
    word2id={'_PAD_': 0}
    path = [opt.train_data_root, opt.validation_data_root, opt.test_data_root]

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()  # strip()删除前导和尾随空格 split()表示根据任何空格进行拆分
                for word in sp[1:]:  # 0位置是标签
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    with open(save_file_path, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')

# build_word2id(opt.word_to_id_path)

def load_word2id(word_to_id_path):
    """

    :param word_to_id_path: word_to_id.txt存放路径
    :return: word_to_id:{word:id}
    """
    word_to_id = {}
    with open(word_to_id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = int(sp[1])
            if word not in word_to_id:
                word_to_id[word] = idx
    return word_to_id

def build_word2vec(fname, word2id, save_to_path=None):
    """

    :param fname: 预训练的word2vec二进制模型
    :param word2id: 语料文本中包含的词汇集
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}
    """
    import gensim
    n_words = max(word2id.values())+1  # 得到词汇表中个数 n_words:58954 59290
    # gensim.models.KeyedVectors.load_word2vec_format导入bin格式
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))  # low,high,size vector_size:50 index_to_key:426677
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]  # word_vecs:58954 59290*50
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

# word2id = load_word2id(opt.word_to_id_path)
# word2vec = build_word2vec(opt.wiki_word2vec_50_bin_path, word2id, opt.corpus_word2vec_path)


def load_corpus_word2vec(corpus_word2vec_path):
    """加载语料库word2vec词向量,相对wiki词向量相对较小"""
    word2vec = []
    with open(corpus_word2vec_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)

def load_corpus(path, word2id, max_len = 50, classes=['0', '1']):
    """

    :param path: 样本语料库的文件
    :param word2ix: 语料文本中包含的词汇集
    :param max_len:
    :param classes:
    :return: 文本内容contents,以及分类标签labels(onehot形式)
    """
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]  # train.txt/validation.txt的每行第一个字符是1/0
            content = [word2id[w] for w in sp[1:]]
            content = content[:max_len]
            if len(content) < max_len:
                content += [word2id['_PAD_']] * (max_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('总样本数为：%d' %(len(labels)))
    print('各个类别样本数如下：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}  # 0:'0' 1:'1'
    labels = np.asarray([cat2id[l] for l in labels])
    return contents, labels