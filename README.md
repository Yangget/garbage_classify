# garbage_classify

-----------------------------
## Introduction

This is a project on garbage sorting. The project comes from ["Huawei Cloud Artificial Intelligence Competition·Garbage Classification Challenge Cup Artificial Intelligence Competition"](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction),
There are a total of forty categories in the data set as follows:
```python
{
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
```
In real life, because of the difference in garbage form, angle of photographing, light, background, etc., the data of the AI ​​training set is difficult to identify the true face of the garbage. Therefore, this competition requires the garbage classification model to have high generalization ability and anti-interference ability to ensure the accuracy of model recognition. 
##  My Work
+ The model I am using is efficientnet-b3,Please visit the repository for details of [it](https://github.com/qubvel/efficientnet).
+ Data enhancement and tricks:
```
# enhancement
width_shift_range=0.05,
height_shift_range=0.05,
horizontal_flip=True,
vertical_flip=True,

# andom_eraser
img = self.eraser(img)

# smooth_labels
y *= 1 - smooth_factor
y += smooth_factor / y.shape[1]

# mixup
X = X1 * X_l + X2 * (1 - X_l)
Y = Y1 * y_l + Y2 * (1 - y_l)

# cosine_decay_with_warmup
```
+ Expand the data set
```shell
补充数据:4175
编号:fimg_
0:一次性快餐盒(236)
1:污损塑料(112)
2:烟蒂(161)
3:牙签(65)
4:破碎花盆及碟碗(70)
5:竹筷(124)
6:剩饭剩菜(73)
7:大骨头(60)
8:果皮(75)
9:烂果肉(93)
10:茶叶渣(87)
11:菜叶(70)
12:蛋壳(119)
13:鱼骨(57)
14:充电宝(91)
15:包(97)
16:化妆品瓶(107)
17:塑料玩具(431)
18:塑料碗盆(100)
19:塑料衣架(179)
20:快递纸袋(58)
21:插头(168)
22:旧衣服(77)
23:易拉罐(106)
24:枕头(106)
25:毛绒玩具(231)
26:洗发水瓶(113)
27:破碎玻璃杯(87)
28:皮鞋(101)
29:砧板(63)
30:纸板箱(67)
31:调料瓶(52)
32:酒瓶(85)
33:金属食品罐(51)
34:锅(125)
35:食用油桶(89)
36:饮料瓶(32)
37:干电池(58)
38:软膏(54)
39:过期药物(50)
```

+ Do experiment (Because the game is not over yet, more details and data are not released for the time being.)

id | inputsize | 翻转 | 平移(py) | otherdatagen | smooth_labels | random_eraser | mixup　| warmuplr | sp | epoch-20 sc |score |　guanfangsc |
--- | ---- | ---| ---| ---- | ---| ---- | --- | --- | ---- | --- | --- | --- |
D0001(baseline) | 300 | F  |  F  |  F  |  F  |  F  |  F  |  Y  |0.15 | 0.9261 | 0.9861 |
D0002           | 300 | F  |  F  |  F  |  F  |  F  |  F  |  Y  |0.15 | 0.9272 | 0.9860 | 0.929825 |
D0003           | 300 | Y  |  F  |  F  |  F  |  F  |  F  |  Y  |0.15 | 0.9265 | 0.9859 | 0.923501 |
D0004           | 300 | Y  |  F  |  F  | 0.1 |  F  |  F  |  Y  |0.15 | 0.9307 | 0.9868 | 0.929009 |
D0005           | 300 | Y  |0.05 |  F  | 0.1 |  F  |  F  |  Y  |0.15 | 0.9374 | 0.9879 |... |
D0006           | 300 | Y  |0.05 |  Y  | 0.1 |  F  |  F  |  Y  |0.15 | 0.9265 | 0.9853 | 0.929213 |
D0008           | 300 | Y  |0.05 |  F  | 0.2 |  F  |  F  |  Y  |0.15 | 0.9339 | 0.9883 | 0.928601 |
D0009           | 300 | Y  |0.05 |  F  | 0.1 |  F  |  F  |  Y  |0.15 | 0.9224 | ... |... |
D0010           | 300 | Y  |0.05 |  F  | 0.1 | 0.2 |  F  |  Y  |0.15 | 0.9367 | 0.9895 |... |
D0011(baseline) | 300 | Y  |0.05 |  F  | 0.1 | 0.2 |  F  |  Y  |0.15 | 0.9272 | 0.9894 | 0.926153 |
D0012(metrics)  | 300 | Y  |0.05 |  F  | 0.1 |  F  |  F  |  Y  |0.15 | 0.9335 | 0.9888 |... |
D0013           | 300 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  Y  |0.15 | 0.9385 | 0.9902 |... |
D0014           | 300 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  Y  |0.15 | 0.9339 | 0.9896 |... |
D0015           | 300 | Y  |0.05 |  F  | 0.1 | 0.2 |  F  |  Y  | 0.1 | 0.9443 | 0.9940 |... |
D0016           | 300 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  F  | 0.1 | 0.9342 |... |... |
D0017(conv5,5)  | 300 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  F  | 0.1 | 0.9305 |... |... |
D0018(eff7)     | 260 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  F  | 0.1 | 0.9342 |... |... |
D0019(efb3Mob)  | 224 | Y  |0.05 |  F  | 0.1 | 0.3 |  F  |  F  | 0.1 | 0.9342 |... |... |
...             | ... | ...| ... | ... | ... | ... | ... | ... | ... | ... | ... |... |

## Use

+ Clone this repository
```shell
git clone https://github.com/Yangget/garbage_classify.git
```
+ Download the [data set](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify_v2.zip) in the same directory
+ Download my extended dataset and the original dataset,The list of supplementary data is as follows:

  The data set is an extension of Huawei's cloud waste classification data. It is sourced from the Internet. The organizer is an undergraduate student at Northeast Forestry University. Please download it in this [repository](ihub.com/Yangget/garbage_classify_expand/tree/master).

+ Run(this code is running locally)
```shell
python run.py --data_url='../datasets/garbage_classify/train_data' --train_url='./model_snapshots' --deploy_script_path='./deploy_scripts'
```
+ Convert to pb
```shell
python run.py --mode=save_pb --deploy_script_path='./deploy_scripts' --freeze_weights_file_path='../model_snapshots/weights_000_0.9811.h5' --num_classes=40
```
+ Evaluation
```shell
python run.py --mode=eval --eval_pb_path='../model_snapshots/model' --test_data_url='../datasets/garbage_classify/train_data'
```

## Impression

The game is coming to an end. Through this competition,as a third-year undergraduate classmate,I have not only enhanced my ability in algorithms and programming, but also have more contact and deeper understanding of image classification algorithms. I also realized the challenge of garbage classification to humans. Thank you very much for this very meaningful game organized by Huawei Cloud.
