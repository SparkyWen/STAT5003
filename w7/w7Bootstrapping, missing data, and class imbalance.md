# Bootstrapping, missing data, and class imbalance

## 1. 知识点

| Bootstrap Type                       | 中文关键词   | 主要用途 / 场景                                              | 抽样思想（大白话）                                           | 典型 R 写法（示意）                                          |
| ------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Nonparametric Bootstrap**          | 非参数自助法 | 估计样本均值、中位数、回归系数等的 SE / CI                   | 把原始样本看作“总体的经验分布”，**有放回地重采样**出 size = n 的样本，重复 B 次 | `boot_mean <- replicate(B, mean(sample(x, replace = TRUE)))` |
| **Parametric Bootstrap**             | 参数自助法   | 模型假设明确（如 (X \sim N(\hat\mu,\hat\sigma^2))），想估计参数的不确定性 | 先拟合一个 parametric model（正态/回归），用**拟合出的分布**生成新的样本，再在每个样本上重新估计参数 | `mu_hat <- mean(x); sd_hat <- sd(x); boot_mu <- replicate(B, mean(rnorm(n, mu_hat, sd_hat)))` |
| **Pairs Bootstrap (for regression)** | 成对自助法   | 线性回归等模型，想对回归系数做 bootstrap；数据可能存在异方差 | 每次从 ((x_i, y_i)) **成对地**有放回抽样 n 个观测，重新拟合模型 | `idx <- sample(1:n, n, replace=TRUE); fit_b <- lm(y[idx] ~ x[idx])` |
| **Residual Bootstrap**               | 残差自助法   | 回归模型，假定误差 i.i.d.、同方差                            | 用拟合的 (\hat y_i) + bootstrap 残差构造新 (y_i^*)；结构固定，只在误差层面自助 | `fit <- lm(y~x); res <- resid(fit); y_star <- fitted(fit) + sample(res, replace=TRUE)` |
| **Block Bootstrap (time series)**    | 分块自助法   | 时间序列/相关数据（观测存在相关性）                          | 把数据切成若干连续小块（blocks），**以块为单位**有放回抽样，保持块内相关结构 | 使用 `tsbootstrap()`(在一些 ts 包) 或自己写循环抽 block      |
| **Wild Bootstrap**                   | wild 自助    | 回归中存在异方差（残差方差随 x 变化）                        | 对残差乘以随机因子（如 ±1），构造新的误差；保留异方差特征    | `res_star <- res * rnorm(n); y_star <- fitted(fit) + res_star` |

| 包 / 函数                                                    | 中文关键词                      | 用途 & 场景                                              | 典型写法（了解即可）                                         |
| ------------------------------------------------------------ | ------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| `rsample::bootstraps()`                                      | tidy 自助样本                   | 在 tidyverse / tidymodels 框架下生成 bootstrap resamples | `r library(rsample) boots <- bootstraps(df, times = 1000) # 每个 split 都可用 analysis( )/assessment( ) ` |
| `caret::train(..., method="lm", trControl=trainControl(method="boot"))` | caret 中的 bootstrap resampling | 用 bootstrap 代替 CV 评估模型性能                        | `r library(caret) ctrl <- trainControl(method="boot", number=1000) fit <- train(y~., data=df, method="lm", trControl=ctrl) ` |

**SMOTE (Synthetic Minority Over-sampling Technique)** tackles class imbalance by creating new **synthetic minority samples** along the line segments between existing minority examples, rather than just duplicating them, to balance the class distribution and reduce overfitting.

## 2. 代码

- **R**

  functions

  - `naniar:: any_na`
  - `naniar:: are_na`
  - `visdat:: vis_dat`
  - `vidat:: vis_miss`
  - `mice::md.pattern`
  - `mice::marginplot`
  - `ROSE::ovun.sample`
  - `themis::step_smote`

## 1. 加载包 & 数据集 Sonar

```
## 1.1 加载需要的 R 包 ------------------------------------

library(mlbench)  # 提供 Sonar 数据集 :contentReference[oaicite:0]{index=0}
library(caret)    # createFolds(), preProcess(), confusionMatrix() 等 :contentReference[oaicite:1]{index=1}
library(MASS)     # 一些统计模型工具（本周主要为完整性）
library(class)    # knn() 分类算法

## 1.2 加载 Sonar 数据 -------------------------------------

data(Sonar, package = "mlbench")  # 加载 Sonar 数据集 :contentReference[oaicite:2]{index=2}

# 查看数据结构（行数、列名、变量类型等）
str(Sonar)
summary(Sonar)

# 说明：
# - 共 208 行观测
# - 60 个数值型特征（声呐强度）
# - 第 61 列 Class 是响应变量: "R" = rock, "M" = metal
```

> 关键词：
>
> - **Sonar**：声呐数据集
> - **Class**：类别（rock / metal，分类任务）
> - **mlbench**：机器学习基准数据集包

------

## 2. 探索性分析：可视化特征分布（为 KNN 做准备）

KNN 非常依赖“距离”，所以各特征要在**相近的数值尺度**上，这也是为什么要看箱线图、再做标准化。Workshop_week7_modified (1)

```
## 2.1 用箱线图查看 60 个特征的分布 ----------------------

set.seed(5003)       # 为了可复现
boxplot(Sonar[ -61 ])  # 去掉第 61 列 Class，只画 60 个数值特征的箱线图
# 提醒：很多特征的量纲/范围不同 -> 必须做标准化(center+scale)
```

> 关键词：
>
> - **Boxplot 箱线图**：看分布形状、是否有 outlier
> - **Scale / Standardization 标准化**：让特征有相似尺度，方便距离计算

------

## 3. 构造训练/测试集，并**正确标准化**

注意：**标准化必须只用训练集算均值/标准差，再用同一套参数变换测试集**，不能把测试集信息“泄露”进去。

### 3.1 把响应 `y` 和特征矩阵 `X` 分开

```
## 3.1 分离响应变量 y 和特征 X ----------------------------

set.seed(5003)

y <- Sonar |>
  dplyr::pull(Class)        # 响应向量 (factor: "R"/"M")
X <- Sonar |>
  dplyr::select(-Class)     # 特征矩阵 (60 列)
n <- length(y)
```

### 3.2 用 `createFolds()` 构造 10-fold 分层抽样

```
## 3.2 创建 10 折 (stratified) 交叉验证分割 ----------------

fold <- createFolds(Sonar$Class, k = 10)  # 返回一个长度为10的 list，每个元素是测试集索引 :contentReference[oaicite:5]{index=5}

str(fold)   # 查看每一折的索引
# fold$Fold01, fold$Fold02, ..., fold$Fold10都是一个整数向量（行号）
```

> 关键词：
>
> - **createFolds**：分层（按类别比例）创建 K 折
> - **Fold01 / Fold02**：第 1/2 折的测试样本行号

### 3.3 选一折做“训练 / 测试拆分 + 标准化 + KNN”示范

```
## 3.3 选定一折做训练/测试划分 ---------------------------

test_id  <- fold$Fold01                # 取第一折作为测试集
train_id <- setdiff(seq_len(n), test_id)  # 其余九折作为训练集

X_train <- X[train_id, ]
X_test  <- X[test_id, ]
y_train <- y[train_id]
y_test  <- y[test_id]

## 3.4 方法一：用 caret::preProcess 来做标准化 ------------

pp <- preProcess(
  X_train,
  method = c("center", "scale")  # 减去均值、除以标准差
)

X_train_scaled <- predict(pp, X_train)  # 用训练集参数标准化训练集
X_test_scaled  <- predict(pp, X_test)   # 用同一参数标准化测试集  :contentReference[oaicite:6]{index=6}
```

> 关键词：
>
> - **preProcess**：预处理（中心化、标准化、BoxCox等）
> - **center / scale**：减均值 / 除标准差（Z-score）

------

## 4. 用 KNN 分类 + 混淆矩阵 & 准确率

### 4.1 拟合 1 次 KNN（k = 5）

```
## 4.1 用 kNN(k = 5) 对测试集分类 -------------------------

knn5 <- knn(
  train = X_train_scaled,  # 训练数据（已标准化）
  test  = X_test_scaled,   # 测试数据（用同样方式标准化）
  cl    = y_train,         # 训练标签
  k     = 5
)

## 4.2 基本混淆矩阵 & 准确率 --------------------------------

table(knn5, y_test)   # 简单的混淆矩阵（预测 vs 真实） :contentReference[oaicite:7]{index=7}

mean(knn5 == y_test)  # 分类准确率 Accuracy
```

### 4.3 用 `confusionMatrix()` 看完整评估指标

```
## 4.3 使用 caret::confusionMatrix 获得完整指标 ------------

cm <- confusionMatrix(knn5, y_test)   # 默认会把"M"当作 positive class :contentReference[oaicite:8]{index=8}
cm

# 重点看输出中的：
# - Accuracy         : 总体准确率
# - 95% CI           : 准确率的95%置信区间
# - Sensitivity      : 对 positive class("M") 的召回率(真阳性率)
# - Specificity      : 对 negative class("R") 的真阴性率
# - Pos Pred Value   : Precision(预测为M中有多少真M)
# - Neg Pred Value   : 预测为R中有多少真R
# - Balanced Accuracy: (Sensitivity + Specificity) / 2
```

> 关键词：
>
> - **Confusion matrix 混淆矩阵**：把预测/真实对比的 2x2 表
> - **Accuracy 准确率**
> - **Sensitivity 灵敏度 / 召回率**
> - **Specificity 特异度**
> - **Precision 精确率**

------

## 5. 手写 10-fold Cross Validation 框架（模板）

虽然讲义里只显示了一部分循环代码，但套路是非常统一的，你复习的时候记住这**范式**就够用了：Workshop_week7_modified (1)

```
## 5.1 初始化要记录的评价指标向量 --------------------------

K <- length(fold)                      # 折数 = 10
acc_vec  <- numeric(K)                 # 存每一折 Accuracy
sens_vec <- numeric(K)                 # 存每一折 Sensitivity
spec_vec <- numeric(K)                 # 存每一折 Specificity

## 5.2 外层循环：对每一折重复“训练 -> 预测 -> 计算指标” -----

for (k in seq_len(K)) {
  test_id  <- fold[[k]]
  train_id <- setdiff(seq_len(n), test_id)
  
  X_train <- X[train_id, ]
  X_test  <- X[test_id, ]
  y_train <- y[train_id]
  y_test  <- y[test_id]
  
  # 标准化：用训练集的均值/标准差
  pp <- preProcess(X_train, method = c("center", "scale"))
  X_train_scaled <- predict(pp, X_train)
  X_test_scaled  <- predict(pp, X_test)
  
  # kNN 分类
  pred <- knn(
    train = X_train_scaled,
    test  = X_test_scaled,
    cl    = y_train,
    k     = 5
  )
  
  # 计算本折的混淆矩阵和指标
  cm <- confusionMatrix(pred, y_test)  # 默认 "M" 为 positive
  acc_vec[k]  <- cm$overall["Accuracy"]
  sens_vec[k] <- cm$byClass["Sensitivity"]
  spec_vec[k] <- cm$byClass["Specificity"]
}

## 5.3 汇总 10 折结果（平均表现） ---------------------------

mean(acc_vec)      # 10-fold 平均准确率
mean(sens_vec)     # 平均灵敏度
mean(spec_vec)     # 平均特异度

sd(acc_vec)        # 也可以看标准差，表示不同折之间波动多大
```

> 关键词：
>
> - **Cross Validation 交叉验证**
> - **10-fold CV**：把数据分成 10 份，轮流做测试集
> - **平均 Accuracy**：估计模型的“泛化误差 (generalisation error)”