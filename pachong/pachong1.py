# -*- coding: utf-8 -*-
"""
 @Time    : 19-8-18 下午1:59
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : fenge.py
"""
import codecs
import os
from glob import glob
import shutil

train_data_dir = './garbage_classify/train_data'
lab = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}
lab_ = lab.values()
label_files = glob(os.path.join(train_data_dir, '*.txt'))
img_paths = []
labels = []
for index, file_path in enumerate(label_files):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline( )
    line_split = line.strip( ).split(', ')
    if len(line_split) != 2:
        print('%s contain error lable' % os.path.basename(file_path))
        continue
    img_name = line_split[0]
    label = int(line_split[1])
    img_paths.append(os.path.join(train_data_dir, img_name))
    labels.append(label)
print("read ok")
for paths, la in zip(img_paths, labels):
    new_path = './newdata/' + str(la)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.copy(paths, new_path)
