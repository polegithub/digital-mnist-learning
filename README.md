
# 数字识别的机器学习算法对比

手写数字识别是机器学习领域最基本的入门内容，这里简单实现了几个不同算法的数字识别实现。
作为机器学习各自算法的demo展示和调参学习。

# 1. 数据源

- 代码：https://github.com/polegithub/digital-mnist-learning
- 数据源：
    在路径`digital-minist-data`下面，训练数据是28*28的图片，标签为对应的数字：0~9.



# 2. 不同算法对比

## 2.1 Logistic Regression 模型
### 2.1.1 代码
Logistic回归是最基本的线性分类模型，也可以用于图像识别，sklearn的代码如下：
```python
"""
lbfgs + l2
"""
parameters = {'penalty': ['l2'], 'C': [2e-2, 4e-2, 8e-2, 12e-2, 2e-1]}
lr_clf = LogisticRegression(
 		penalty='l2', solver='lbfgs', multi_class='multinomial', 	 max_iter=800,  C=0.2)
 	

"""
liblinear + l1
"""
parameters = {'penalty': ['l1'], 'C': [2e0, 2e1, 2e2]}
lr_clf=LogisticRegression(penalty='l1', multi_class='ovr', max_iter=800,  C=4 )

```
### 2.1.2 `liblinear + L1`测试结果
样本数：1000
```python
grid_scores_:
mean score | scores.std() * 2 | params
0.826      | (+/-0.035)       | {'C': 2.0, 'penalty': 'l1'}
0.820      | (+/-0.050)       | {'C': 200.0, 'penalty': 'l1'}
0.819      | (+/-0.031)       | {'C': 20.0, 'penalty': 'l1'}
```

### 超参数结论： 
1. C越大，均方差越大
2. 不同的 C 对 mean score 差异不大
3. 1000的样本数太少，仅供参考


### 2.1.3 `lbfgs + L2`测试结果
样本数：1000
```python
grid_scores_:
mean score | scores.std() * 2 | params
0.850      | (+/-0.036)       | {'C': 0.12, 'penalty': 'l2'}
0.848      | (+/-0.034)       | {'C': 0.08, 'penalty': 'l2'}
0.848      | (+/-0.034)       | {'C': 0.2, 'penalty': 'l2'}
0.844      | (+/-0.045)       | {'C': 0.04, 'penalty': 'l2'}
0.839      | (+/-0.055)       | {'C': 0.02, 'penalty': 'l2'}
```

### 超参数结论： 
1. 整理来看，无太大差异，相对来说，C 的取值 0.12 表现稍微好一点
2. 1000的样本数太少，仅供参考



## 2.2 KNN 近邻模型

### 2.2.1 代码
K最近邻(kNN，k-NearestNeighbor)分类算法也是很基本的算法。识别图像的算法有{'auto', 'ball_tree', 'kd_tree', 'brute'} ，本质是通过计算距离来计算矩阵之间的相似度。  sklearn的代码如下：
```
knn_clf = KNeighborsClassifier(
    n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)
```
### 2.2.2 测试结果
样本数：1000

```python
grid_scores_:
mean score | scores.std() * 2 | params
0.850      | (+/-0.024)       | {'algorithm': 'kd_tree', 'n_neighbors': 3}
0.850      | (+/-0.024)       | {'algorithm': 'ball_tree', 'n_neighbors': 3}
0.846      | (+/-0.008)       | {'algorithm': 'kd_tree', 'n_neighbors': 7}
0.846      | (+/-0.008)       | {'algorithm': 'ball_tree', 'n_neighbors': 7}
0.843      | (+/-0.033)       | {'algorithm': 'kd_tree', 'n_neighbors': 5}
0.843      | (+/-0.033)       | {'algorithm': 'ball_tree', 'n_neighbors': 5}
0.832      | (+/-0.020)       | {'algorithm': 'kd_tree', 'n_neighbors': 9}
0.832      | (+/-0.020)       | {'algorithm': 'ball_tree', 'n_neighbors': 9}
```
### 超参数结论： 
1. 不同算法 (kd_tree / ball_tree) 对结果无影响，ball_tree 只是优化了维度灾难的问题
2. n_neighbors 目前结果来看，3 的 mean score最佳，但是 7 的均方差最小。
3. 1000的样本数太少，仅供参考



## 2.3 RandomForest 随机森林模型

### 2.3.1 代码

```python
parameters = {'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 12, 100]}

rf_clf = RandomForestClassifier(n_estimators=400, n_jobs=4, verbose=1)
```

### 2.3.2 测试结果
样本数：1000

```python
grid_scores_:
mean score | scores.std() * 2 | params
0.877      | (+/-0.020)       | {'criterion': 'gini', 'max_features': 12}
0.876      | (+/-0.023)       | {'criterion': 'entropy', 'max_features': 12}
0.875      | (+/-0.025)       | {'criterion': 'gini', 'max_features': 'auto'}
0.871      | (+/-0.045)       | {'criterion': 'gini', 'max_features': 100}
0.869      | (+/-0.034)       | {'criterion': 'entropy', 'max_features': 100}
0.866      | (+/-0.025)       | {'criterion': 'entropy', 'max_features': 'auto'}

```
### 超参数结论： 
1. max_features 目前最佳为 12
2. gini 略优于 entropy, 但并不明显, 其实各项参数的结果都比较接近。 
3. 1000的样本数太少，仅供参考



## 2.4 SVM 支持向量机模型
支持向量机SVM是通过构建决策面实现分类的模型，决策面构建的前提是选取的样本点(支持向量)距离决策面的距离最大。具体在求解最大距离的时候，是通过拉格朗日乘子法进行计算的。
同时由于很多数据并不是线性的，所以在求解距离的时候会引入一个松弛变量$\varepsilon_i$, 这个$\varepsilon_i$求和后的参数是C，C被称作复杂性参数，表达了对错误的容忍度：
- 默认为1，C较大表示对错误的容忍度越低。
- 一般需要通过交叉验证来选择一个合适的C
- 一般来说，如果噪音点较多时，C需要小一些。

scikit-learn中SVM的算法库分为两类，一类是分类的算法库，包括SVC， NuSVC，和LinearSVC 3个类。另一类是回归算法库，包括SVR， NuSVR，和LinearSVR 3个类。



> 关于NuSVC的说明：
> nuSVC使用了nu这个等价的参数控制错误率，就没有使用C，为什么我们nuSVR仍然有这个参数呢，不是重复了吗？这里的原因在回归模型里面，我们除了惩罚系数C还有还有一个距离误差ϵϵ来控制损失度量，因此仅仅一个nu不能等同于C.也就是说回归错误率是惩罚系数C和距离误差ϵϵ共同作用的结果。
> 来源：https://www.cnblogs.com/pinard/p/6117515.html



### 2.4.1 代码
```python
# nuSVC
parameters = {'nu': (0.5, 0.02, 0.01), 'gamma': [0.02, 0.01,'auto'],'kernel': ['rbf','sigmoid']}
svc_clf = NuSVC(nu=0.1, kernel='rbf', verbose=0)

# SVC
parameters = {'gamma': (0.05, 0.02, 'auto'), 'C': [10, 100, 1.0], 'kernel': ['rbf','sigmoid']}
svc_clf = SVC(gamma=0.02)

```

### 2.4.2 NuSVC 测试结果
样本数：1000

```python
grid_scores_:
mean score | scores.std() * 2 | params
0.902      | (+/-0.017)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.01}
0.901      | (+/-0.016)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.02}
0.896      | (+/-0.027)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.02}
0.896      | (+/-0.027)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.01}
0.888      | (+/-0.040)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.5}
0.879      | (+/-0.031)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.5}
0.874      | (+/-0.024)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.02}
0.872      | (+/-0.019)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.01}
0.859      | (+/-0.041)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.5}
0.857      | (+/-0.032)       | {'gamma': 'auto', 'kernel': 'sigmoid', 'nu': 0.02}
0.856      | (+/-0.042)       | {'gamma': 'auto', 'kernel': 'sigmoid', 'nu': 0.5}

```
### 超参数结论： 
1. rbf 优于 sigmoid
2. gamma: auto的效果并不好, 结果来看最佳为 0.02
3. nu为 0.5 时效果并不好，0.01 和 0.02 时无明显优势，可以考虑加入 0.05 对比测试
4. 1000的样本数太少，仅供参考



### 2.4.3 SVC 测试结果

样本数：1000

```python
grid_scores_:
mean score | scores.std() * 2 | params
0.901      | (+/-0.016)       | {'C': 10, 'gamma': 0.02, 'kernel': 'rbf'}
0.901      | (+/-0.016)       | {'C': 50, 'gamma': 0.02, 'kernel': 'rbf'}
0.894      | (+/-0.031)       | {'C': 1.0, 'gamma': 0.02, 'kernel': 'rbf'}
0.883      | (+/-0.026)       | {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
0.874      | (+/-0.026)       | {'C': 50, 'gamma': 'auto', 'kernel': 'rbf'}
0.870      | (+/-0.033)       | {'C': 50, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.866      | (+/-0.020)       | {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.864      | (+/-0.019)       | {'C': 10, 'gamma': 0.05, 'kernel': 'rbf'}
0.864      | (+/-0.019)       | {'C': 50, 'gamma': 0.05, 'kernel': 'rbf'}
0.856      | (+/-0.014)       | {'C': 1.0, 'gamma': 0.05, 'kernel': 'rbf'}
0.822      | (+/-0.057)       | {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'}
0.754      | (+/-0.053)       | {'C': 1.0, 'gamma': 0.02, 'kernel': 'sigmoid'}

```
### 超参数结论： 
1. rbf 优于 sigmoid
2. gamma: auto的效果并不好, 结果来看score最佳为 0.02
3. C取10和50甚至100，对mean score无明显影响，但C取1的时候，均方差偏大。
4. 1000的样本数太少，仅供参考



## 2.5 CNN 卷积神经网络模型

CNN在图像识别中曾经放光彩，毫无疑问相比前面的几个算法，必然是准确率更高的。数字识别在很多框架中作为基础example，所以这里不再进一步展开，直接用了keras的一个例子，源码见这里：[mnist_cnn.py](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
### 2.5.1 代码
```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

...

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
activation='relu',
input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])
model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

### 2.5.2 测试结论
```python
 3840/60000 [>.............................] - ETA: 1:36 - loss: 1.3122 - acc: 0.5766
 3968/60000 [>.............................] - ETA: 1:35 - loss: 1.2842 - acc: 0.5862
 4224/60000 [=>............................] - ETA: 1:34 - loss: 1.2694 - acc: 0.5909
 4480/60000 [=>............................] - ETA: 1:34 - loss: 1.2335 - acc: 0.6029
25472/60000 [===========>..................] - ETA: 56s - loss: 0.4524 - acc: 0.8605
...
59520/60000 [============================>.] - ETA: 0s - loss: 0.0462 - acc: 0.9860
59904/60000 [============================>.] - ETA: 0s - loss: 0.0461 - acc: 0.9860
60000/60000 [==============================] - 98s 2ms/step - loss: 0.0461 - acc: 0.9860 - 
    
val_loss: 0.0280 - val_acc: 0.9908
```



# 3. 结论

具体的参数结论都在每个算法各自的测试结果里，整体横向对比，除了CNN效果最好，其他几个算法中SVM的效果是仅次于CNN的。在神经网络出现之前，SVM一直是每年论文投递的热门领域，但知道deep learning的出现，准确率大幅上升之后，SVM的使用者大大减少。很多学者的研究方向转向的RNN,CNN,LSTM以及各种新模型等。诚然，神经网络的效果的显著的，但是很多基础的算法还是需要掌握和了解，因为真实的场景中，并不是每个公司都有足够量级的数据供deep learning去训练。



Reference
-----

1. [数字识别，从KNN,LR,SVM,RF到深度学习 - 腾讯云](https://cloud.tencent.com/developer/article/1059210)
2. [基于SVM的思想做CIFAR-10图像分类](https://cloud.tencent.com/developer/article/1386846)
3. [scikit-learn SVM算法库使用概述, 刘建平Pinard](https://www.cnblogs.com/pinard/p/6117515.html)



