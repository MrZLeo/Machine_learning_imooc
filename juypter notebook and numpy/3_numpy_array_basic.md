# Numpy-Array-Basic

### Why we choose Numpy:

#### Python list 的特点
* 数据类型不限 **-->** 灵活性强 || 效率降低 

#### array.array 的特点
* **单一类型数据** ，弥补了原生list的不足
* **没有把数据当作向量或矩阵，不支持基本运算**
* 不支持 **float -> int** 的隐性转换

#### numpy.array 的特点
* **单一数据类型**
* 多种操作指令
* 丰富的矩阵运算

### numpy 的使用：
#### 1. create：
##### 直接创建:
```python
import numpy as np
nparr = np.array( [ i for i in range(10) ] )
```
##### 创建特殊矩阵：
###### 1. 零矩阵 **zeros**
```python 
np.zeros( shape = (3, 5), dtype = int )
```
###### 2. 全1矩阵 **ones**
```python
np.ones(10)
```
###### 3. 全部为指定数字 **full**
```python
np.full(shape =( (3, 5), fill_value = 666)
```
###### 4. **arrange**
* in python: `[i for i in range(0, 1, 0.2)]` 
    * 第一个数字：左区间（闭） 
    * 第二个数字：右区间（开）
    * 第三个数字：步长
    * 特点：步长为整数
* in numpy: `np.arrage(0， 1， 0.2)`
    * **特点** ：步长可为浮点数
    
###### 5. **linspace**
* `np.linspace(0, 20, 10)`
    * 第三位数字表示在所给区间中平均分为x个数
    * 左右区间都是 **闭区间**

###### 6. **random**
* **调用** ：`np.random.xxx`

1. **randint** `(0, 10)`
    * [0, 10) 之间的随机数
    * size = （矩阵的大小）
         
2. seed
    * 决定随机数的种子
    * 测试算法的时候可以使用固定的种子
        
3. random
    * [0, 1)之间的随机数
    * 可以给出size定义所需要的矩阵
4. normal
    * 符合正态分布的随机数
    * `np.random.normal(a, b, size)`
        * a -> 期望
        * b -> 方差

#### 2. operations 操作：

##### 1. numpy.array 的基本属性
1. **ndim**
    * 矩阵的维度
2. **shape**
    * 矩阵的尺寸大小
3. **size**
    * 矩阵的元素个数
4. **dtype**
    * 显示array中的元素的数据类型


##### 2. numpy.array 的数据访问
1. 下标索引
* `x[0]`
* `x[a, b]` -> row a+1, column b+1
        
2. **-1** 实现倒序访问：
* `x[-1]` -> 最后一个元素
        
3. 切片：
* 前一个数字默认从头开始
    * `x[:5]`  -> [0, 5)个元素
* 后一个数字默认最后一位数
    * `x[0:]`  -> all 

##### 3. **subarray 子阵列** 
 * 切片产生的子阵列是与原来的矩阵  **共用存储空间的变量** ，**一旦两者其一的数据改变，会导致另一方数据的变化**
 * **`.copy()`** 产生副本可以避免问题

##### 4. **reshape**
* `.reshape()` -> 将原来的向量变成指定大小的矩阵

##### 5. concatenate and split 合并与分割

###### 1. concatenate
* 向量的合并：直接并到原向量最后
* 矩阵的合并：默认第一维度合并
    * **axis** ：指定合并的维度，默认axis = 0
    * 合并时在**合并维度**上两矩阵大小要相同
    * 可以用reshape进行变换后再进行合并

###### 2. _ stack
* `np.vstack`  ->第一维度合并
* `np.hstack`  ->第二维度合并

###### 3. split
```python
x =np.arrage(10)   
x1, x2, x3 = np.split(x, [3, 7])

--> x1[0, 1, 2]
--> x2[3, 4, 5 ,6]
--> x3[7, 8, 9]
```
* 第二个变量以数组形式给出分割点，为不可取点
* 若为矩阵分割：
    * 默认axis = 0 ->沿着第一维度方向分割

###### 4. _ split
* vstack -> 第一维度
* hstack -> 第二维度

#### 3. computation 运算

##### 点乘
* python原生矩阵点乘的语法得到的是 **list的复制**
* numpy的点乘得到的是数学意义上的结果
* numpy的效率优化使得相同代码运行速度比原生语法快**两个数量级**

##### NumPy’s UFuncs (Universal Functions)
1. 加法 +
2. 减法 -
3. 乘法 *
4. 除法 /
5. 整数除法 //
6. 乘方 **
7. 取余 %
8. 取倒数 1 / x
9. 绝对值 `np.abs()`
10. 三角函数与反三角函数 `np.sin/cos/tan()`
`np.arcsin()`
11. 指数:
* `np.exp()` -> 以e为底
* `np.exp2()` -> 以2为底
*  任意底数：`np.power()`
12. 对数：
* 以e为底：`np.log()`
* 以2为底：`np.log2()`
* 以10为底：`np.log10()` 

##### 矩阵运算
* 基本四则运算 -> 对应元素进行运算
* 乘法：
    * 单写 A * B --> 对应元素相乘
    * 矩阵相乘 ：`A.dot(B)` -->A×B
    * 这个特性对于机器学习的数据处理非常重要
* 转置：`A.T`
* 伪逆：`np.linalg.pinv`
    * **伪逆** ：X.dot(pinvX) == I

#### 4. aggregation 聚合

#####  sum
* `sum()` 
* 原生和numpy都有相应的函数，但numpy经过优化，效率是原生的两个数量级

##### min, max
* `np.min`
* `np.max`
* 求出矩阵中的最大值和最小值

##### 多维度聚合
* 给 **`sum()`** 增加参数axis
* axis表示**压缩的维度**
* for instance：**axis == 0** 时，朝第一维度方向，把每列的元素相加

##### 其他聚合操作

###### `np.prod`  **求乘积**
###### `np.mean` **均值**
###### `np.median` **中位数**
###### `np.percentile（q = ?）` 
* **求百分之?的位数（即百分之多少的数与其他数的分界线，从小到大排列）**
* 我们关心的百分位数通常有：
    * 25%， 50%， 75%， 100%
###### `np.var` **方差**
###### `np.std` **标准差**

#### 5. Arg 索引

##### 索引的基本运用

###### `np.argmin` **获得最小值的索引**
###### `np.argmax` **获得最大值的索引**

##### 排序与索引

###### `np.random.shuffle（X）` **随机乱序**
###### `np.sort` **排序**
###### `np.argsort` **按照值的排序输出对应值的索引**
###### `np.partition` 
* **按所给值进行分区，并不进行具体排序**
###### `np.argpartition`
* **如上同理，输出对排序后的值的原位置索引**


#### Comparison and Fancy Indexing

##### Fancy Indexing
###### x[a]
* 取出x中第a+1个元素
###### x[a, b] -->切片
* 取出x中[a+1, b+1)个元素
###### x[a, b, c]
* 切片且步长为c
###### 索引搜索
```python
ind = [3, 5, 7]
x[ind]
```
* 取出相应索引的元素
* 还可以是二维数组，多维矩阵......

##### Fancy Indexing 与二维数组
* 可以直接输入相应的数组作为索引
```python
col = np.array([1, 2, 3])
X[:2, col]
```
* col 还可以是bool型的数组：
```python
col = [True, False, True, False]
X[0, col]
```
##### 比较
###### 数组自身的比较：
1. < and >
2. ==
3. !=
4. 含有数组的方程的不等式运算

* **warning**
比较后得到一个原数组shape的bool类型矩阵

###### 使用比较的结果：
* `np.count_nonzero(X >= 3)`
* `np.sum(X >= 3)`
    * bool类型数组中只有true才是非零，所以这个函数可以用来统计数组中满足条件的元素个数

* `np.any(X == 0)`
    * 如果有满足条件的元素存在返回True

* `np.all(X >= 0)`
    * 如果所有元素都满足条件 --> True

* **warning**
    * 在该类函数中运用**非**要使用 **~**

##### 比较结果和Fancy Indexing
* 比较结果是一个bool类型的数组
* 把比较结果的数组输入进原数组中，可以得到满足比较条件的元素值
* 还可以复合使用切片的方法