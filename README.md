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
