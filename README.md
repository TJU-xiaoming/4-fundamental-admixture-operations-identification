# 4-fundamental-admixture-operations-identification
The repository consists of two projects. a)In the picture is the conventional   operation of addition subtraction and multiplication operations 2)The picture contains two or three lines of formulas, assignment and operation, respectively.
# 简单四则混合运算识别
 
## 问题描述

第一个项目的目的是为了解决一个 OCR 问题，实现图像——>文字的转换过程。

### 数据集描述

一共包含10万张180*60的图片和一个labels.txt的文本文件。每张图片包含一个数学运算式，运算式包含：

3个运算数：3个0到9的整型数字；
2个运算符：可以是+、-、*，分别代表加法、减法、乘法
0或1对括号：括号可能是0对或者1对

图片的名称从0.png到99999.png

文本文件 labels.txt 包含10w行文本，每行文本包含每张图片对应的公式以及公式的计算结果，公式和计算结果之间空格分开，例如图片中的示例图片对应的文本如下所示：

```
(3-7)+5 1
5-6+2 1
(6+7)*2 26
(4+2)+7 13
(6*4)*4 96
```

### 评价指标

评价指标是准确率，所以要求序列与运算结果都正确才会判定为正确。

我们除了会使用准确率作为评估标准以外，还会使用 CTC loss 来评估模型。

## 使用 captcha 进行数据增强

原始数据集提供了10万张图片，我们可以通过Captcha，参照原始数据集中的图片，随机生成更多数据，进而提高准确性。根据题目要求，label 必定是三个数字，两个运算符，一对或没有括号，根据括号规则，只有可能是没括号，左括号和右括号，因此很容易就可以写出数据生成器的代码。

### 生成器

除了生成算式以外，还有一个值得注意的地方就是初赛所有的减号（也就是“-”）都是细的，但是我们直接用 captcha 库生成图像会得到粗的减号，所以我们修改了 [image.py] 中的代码，在 `_draw_character` 函数中我们增加了一句判断，如果是减号，我们就不进行 resize 操作，这样就能防止减号变粗，使用上面的生成器，生成更多的数据用于训练。

## 模型结构
两个双向GRU
## 模型训练

在经过之前的几次尝试以后，我发现在有生成器的情况下，训练代数越多越好，因此直接用 adam 跑了50代，每代10万样本，可以看到模型在10代以后基本已经收敛，计算最后的 ctc loss，进而训练模型。

## 总结

这个项目是非常简单的，因此我们才能得到这么准的分数，之后进一步提升了难度，将测试集提高到了20万张，可行的改进方法是将准确率进一步降低，充分训练模型，将多个模型结果融合等。

### 官方扩充测试集的难点

在扩充数据集上，我们发现有一些图片预测出来无法计算，比如 `[629,2271,6579,17416,71857,77631,95303,102187,117422,142660,183693]` 等，但是经过一定的图像处理（转成灰度图，然后直方图均值化），我们可以显现出来它的真实面貌。

```py
IMAGE_DIR = 'image_contest_level_1_validate'
index = 117422

img = cv2.imread('%s/%d.png' % (IMAGE_DIR, index))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = cv2.equalizeHist(gray)
```

# 多行赋值四则混合运算识别

## 问题描述

每张图片中有两到三行式子，分别是赋值式与运算式，最终对文本序列进行识别。

### 数据集描述

该项目数据集一共包含10万张图片和一个labels.txt的文本文件。每张图片包含一个数学运算式，运算式中包含：

1. 图片大小不固定
2. 图片中的某一块区域为公式部分
3. 图片中包含二行或者三行的公式
4. 公式类型有两种：赋值和四则运算的公式。两行的包括由一个赋值公式和一个计算公式，三行的包括两个赋值公式和一个计算公式。加号（+） 即使旋转为 x ，仍为加号， * 是乘号
5. 赋值类的公式，变量名为一个汉字。 汉字来自两句诗（不包括逗号）： 君不见，黄河之水天上来，奔流到海不复回 烟锁池塘柳，深圳铁板烧
6. 四则运算的公式包括加法、减法、乘法、分数、括号。 其中的数字为多位数字，汉字为变量，由上面的语句赋值。
7. 输出结果的格式为：图片中的公式，一个英文空格，计算结果。 其中： 不同行公式之间使用英文分号分隔 计算结果时，分数按照浮点数计算，计算结果误差不超过0.01，视为正确。
8. 整个label.txe文件使用UTF8编码,具体内容如下：

'''
流=42072;圳=86;(圳-(97510*45921))*流/35864	-5.252849e+09
回=38093;铁=50521;铁*(4560-64206-回/47726)	-3.013416e+09
到=37808;(10220+到/78589)*(70612*88431)	6.381965e+13
不=87863;42263*57806-不/76028*38980	2.443010e+09
到=94310;锁=61045;((63526+锁)-21038)*到/81905	1.192137e+05
         .
         .
         .
'''
### 评价指标

只考虑文本序列的识别，不评价运算结果。
而我们本地除了会使用准确率作为评估标准以外，还会使用 CTC loss 来评估模型。

### 进行数据分析

先对数据集中的数据进行分析，才可以定义生成器。

### 数据预处理

由于原始的图像十分巨大，直接输入到 CNN 中会有90%以上的区域是没有用的，所以我们需要对图像做预处理，裁剪出有用的部分。然后因为图像有两到三个式子，因此我们采取的方案是从左至右拼接在一起，这样的好处是图像比较小。

主要使用了以下几种技术：

转灰度图->直方图均衡化->中值滤波->开闭运算->二值化->关键区域轮廓查找->边界矩形框处轮廓->微调->三个式子横向连接

使用直方图均衡：提高对比度
使用中值滤波：过滤噪点与干扰线
使用开闭运算：进行形态学变换（滤波）
使用轮廓查找：初定三个框的位置
使用边界矩形：函数得到矩形的 (x, y, w, h)，完成关键区域提取，提取之后将绿色的矩形画在了原图上。

## 模型结构
两个双向GRU
CNN 的结构由原来的两层卷积一层池化，改为了多层卷积，一层池化的结构，由于卷积层分别是3，4和6层，简称为 346 结构。

模型思路是这样的：首先输入一张图，然后通过 cnn 导出 (112, 10, 128) 的特征图，其中112就是输入到 rnn 的序列长度，10 指的是每一条特征的高度是10像素，将后面 (10, 128) 的特征合并成1280，然后经过一个全连接降维到128维，就得到了 (112, 128) 的特征，输入到 RNN 中，然后经过两层双向 GRU 输出112个字的概率，然后用 CTC loss 去优化模型，得到能够准确识别字符序列的模型。

### 其他参数

相比第一个项目的模型，这里进行了一些修改：

* padding 变为了 same，不然我觉得特征图的高度不够，无法识别分数
* 增加了 l2 正则化，loss loss 变得更大了，但是准确率变得更高了（添加 l2 的部分包括卷积层的 kernel，BN 层的 gamma 和 beta，以及全连接层的 weights 和 bias）
* 各个层的初始化变为了 he_uniform，效果比之前好
* 去掉了 dropout，不清楚影响如何，但是反正有生成器，应该不会出现过拟合的情况

## 生成器

为了得到更多的数据，提高模型的泛化能力，使用了一种很简单的数据扩充办法，那就是根据表达式中的中文随机挑选赋值式，组成新的样本。这里我们取了前 350*256=89600 个样本来生成，用之后的 10240 个样本来做验证集，还有一点零头因为太少就没有用了。

导入数据的时候，先读取运算式的图像，然后按中文导入赋值式的图像到字典中。因为字典中的 key 是无序的，所以我们在字典中存的是 list，列表是有序的。

```py
from collections import defaultdict

cn_imgs = defaultdict(list)
cn_labels = defaultdict(list)
ss_imgs = []
ss_labels = []

for i in tqdm(range(n1)):
    ss = df[0][i].decode('utf-8').split(';')
    m = len(ss)-1
    ss_labels.append(ss[-1])
    ss_imgs.append(cv2.imread('crop_split2/%d_%d.png'%(i, 0)).transpose(1, 0, 2))
    for j in range(m):
        cn_labels[ss[j][0]].append(ss[j])
        cn_imgs[ss[j][0]].append(cv2.imread('crop_split2/%d_%d.png'%(i, m-j)).transpose(1, 0, 2))
```

然后实现生成器，这里继承了 keras 里的 Sequence 类：

```py
from keras.utils import Sequence

class SGen(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.X_gen = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        self.y_gen = np.zeros((batch_size, n_len), dtype=np.uint8)
        self.input_length = np.ones(batch_size)*rnn_length
        self.label_length = np.ones(batch_size)*38
    
    def __len__(self):
        return 350*256 // self.batch_size
    
    def __getitem__(self, idx):
        self.X_gen[:] = 0
        for i in range(self.batch_size):
            try:
                random_index = random.randint(0, n1-1)
                cls = []
                ss = ss_labels[random_index]
                cs = re.findall(ur'[\u4e00-\u9fff]', df[0][random_index].decode('utf-8').split(';')[-1])
                random.shuffle(cs)
                x = 0
                for c in cs:
                    random_index2 = random.randint(0, len(cn_labels[c])-1)
                    cls.append(cn_labels[c][random_index2])
                    img = cn_imgs[c][random_index2]
                    w, h, _ = img.shape
                    self.X_gen[i, x:x+w, :h] = img
                    x += w+2
                img = ss_imgs[random_index]
                w, h, _ = img.shape
                self.X_gen[i, x:x+w, :h] = img
                cls.append(ss)

                random_str = u';'.join(cls)
                self.y_gen[i,:len(random_str)] = [characters.find(x) for x in random_str]
                self.y_gen[i,len(random_str):] = n_class-1
                self.label_length[i] = len(random_str)
            except:
                pass
        
        return [self.X_gen, self.y_gen, self.input_length, self.label_length], np.ones(self.batch_size)
```

首先随机取一个表达式，然后用正则表达式找里面的中文，再从{中文：图像数组}的字典中随机取图像，经过之前预处理的方式拼接成一个新的序列。
比如随机取了一个 `85882*(河/76020-37023)-铁`，然后我们从铁的赋值式中随机取一个，再从河的赋值式中随便取一个，拼起来就行。
可以看到背景颜色是不同的，但是并不影响模型去识别。

## 模型训练

方法:1：先用 Adam(1e-3) 学习率快速收敛50代，然后用 Adam(1e-4) 跑50代，达到一个不错的 loss，最后用 Adam(1e-5)微调50代，每一代都保存权值，并且把验证集的准确率跑出来。

方法2：
先用Adam(1e-3)的学习率训练20代，然后 1e-4 和 1e-5 交替训练2次，每次训练取验证集 loss 最低的结果继续训练，虽然速度快，但是准确率不够好。

方法3：将全部训练集都用于训练。

## 预测结果

读取测试集的样本，然后用 `base_model` 进行预测。

```py
X = np.zeros((n, width, height, channels), dtype=np.uint8)

for i in tqdm(range(n)):
    img = cv2.imread('crop_split2_test/%d.png'%i).transpose(1, 0, 2)
    a, b, _ = img.shape
    X[i, :a, :b] = img

base_model = load_model('model_346_split2_3_%s.h5' % z)
base_model2 = make_parallel(base_model, 4)

y_pred = base_model2.predict(X, batch_size=500, verbose=1)
out = K.get_value(K.ctc_decode(y_pred[:,2:], input_length=np.ones(y_pred.shape[0])*rnn_length)[0][0])[:, :n_len]
```

对于生成的数据，输出到文件的部分有一点值得一提，就是如何计算出真实值：

```py
ss = map(decode, out)

vals = []
errs = []
errsid = []
for i in tqdm(range(100000)):
    val = ''
    try:
        a = ss[i].split(';')
        s = a[-1]
        for x in a[:-1]:
            x, c = x.split('=')
            s = s.replace(x, c+'.0')
        val = '%.2f' % eval(s)
    except:
#         disp3(i)
        errs.append(ss[i])
        errsid.append(i)
        ss[i] = ''
    
    vals.append(val)
    
with open('result_%s.txt' % z, 'w') as f:
    f.write('\n'.join(map(' '.join, list(zip(ss, vals)))).encode('utf-8'))
    
print len(errs)
print 1-len(errs)/100000.

# output
22
0.99978
```

其中的思路说起来也很简单，就是将表达式中的赋值式中文替换为赋值式的数字，然后直接用 python eval 得到结果，算不出来的直接留空即可。这个0.9977模型的可算率达到了0.99978，也就是说十万个样本里面只有22个样本不可算，当然，实际上还是有一些样本即使可算，也会因为各种原因识别错，比如5和6就是错误的重灾区，某些数字被干扰线切过，导致肉眼都辨认不清等。

## 模型结果融合

模型结果融合的规则很简单，对所有的结果进行次数统计，先去掉空的结果，然后取最高次数的结果即可，其实就是简单的投票。

```py
import glob
import numpy as np
from collections import Counter

def fun(x):
    c = Counter(x)
    c[' '] = 0
    return c.most_common()[0][0]

ss = [open(fname, 'r').read().split('\n') for fname in glob.glob('result_model*.txt')]
s = np.array(ss).T
with open('result.txt', 'w') as f:
    f.write('\n'.join(map(fun, s)))
```

将上面 loss 图中的三个模型结果融合以后，最后得到了0.99868的测试集准确率。


