# @Time : 2021/6/29 3:35 PM 
# @Author : zyc
# @File : draw.py 
# @Title :
# @Description :

import matplotlib.pyplot as plt
from config import opt
import time

def draw_process(title, iters, costs, label_cost, save_path ):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.legend()
    plt.grid()
    # 使用plt保存loss图像
    loss_prefix = save_path +"_"
    loss_path = time.strftime(loss_prefix + "%m%d_%H:%M:%S.png")
    plt.savefig(loss_path)
    plt.show()