# @Time : 2021/6/25 11:00 AM 
# @Author : zyc
# @File : BasicModule.py 
# @Title :
# @Description :

import time
import torch as t
import os

class BasicModule(t.nn.Module):
    print('BasicModule:', os.getcwd())
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path):
        """
        加载指定路径的模型
        :param name:模型地址
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，使用"模型名称+时间"作为文件名
        如 AlexNet_0710_23:57:29.pth
        :param name:模型名称
        :return:
        """
        if name is None:
            prefix = os.getcwd() + '/checkpoints/' + self.module_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name