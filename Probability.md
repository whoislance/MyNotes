[TOC]

# 第一章 随机事件与概率

### 1.2 概率的定义

### 1.3 概率的性质

关于古典概率，先跳过。

### 1.4 条件概率

- 乘法公式
  $$
  P(AB)=P(B)P(A|B)
  $$

- 全概率公式
  $$
  P(A)=\sum_{t=1}^n P(B_i)P(A|B_i)
  $$
  
- 贝叶斯公式
  $$
  P(B|A)=\frac{P(B)P(A|B)}{P(A)} \\
  P(B_i|A)=\frac{P(B_i)P(A|B_i)}{P(A)}=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^nP(B_j)P(A|B_j)}
  $$
  P(B)是B的先验概率

  P(B|A)是B的后验概率

### 1.5 独立性

- 两个事件的独立性

  如果下式陈粒，则事件A与B相互独立：
  $$
  P(AB)=P(A)P(B)
  $$



# 第二章 随机变量及其分布

### 2.1 随机变量及其分布

- 分布函数 CDF

  设$X$是一个随机变量，对任意实数x，称
  $$
  F(x)=P(X\le x)
  $$
  为随机变量X的分布函数，称X服从F(x)，记为$X\sim F(x)$

- 概率密度函数 PDF

  设随机变量$X$的分布函数为F(x)，如果存在实数轴上的一个非负可积函数p(x)，似的对任意实数$x$有
  $$
  F(x)=\int_{-\infin}^x p(t)dt
  $$
  称P(x)为X的概率密度函数
  $$
  F^{'}(x) = p(x)
  $$
  

### 2.2 随机变量的数学期望

- 数学期望

  - 离散
    $$
    E(X)=\sum_{i=1}^{\infin}x_ip(x_i)
    $$
  
- 连续
    $$
    E(X)=\int_{-\infin}^\infin xp(x)dx
    $$
  
- 性质
    $$
    E(aX)=aE(X)
    $$
### 2.3 随机变量的方差与标准差

- 方差
  $$
  Var(X)=E(X-E(X))^2
  $$

- 标准差
  $$
  \sigma(X)= \sqrt{Var(X)}
  $$

- 性质
  $$
  Var(X)=E(X^2)-[E(X)]^2
  $$

- 切比雪夫不等式
  $$
  P(|X-E(X)|\ge\epsilon)\le\frac{Var(X)}{\epsilon^2}
  $$

### 2.4 常用离散分布

- 二项分布
  $$
  P(X=k) = {n\choose k}p^k(1-p^{n-k})
  $$

  - 期望
    
    $E(X)=np$
    
  - 方差
    
    $Var(X)=np(1-p)$
  
- 泊松分布
  $$
  P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}
  $$

    - 期望
  
      $E(X)=\lambda$
  
  - 方差
  
    $Var(X)=\lambda$

- 泊松定理

  在n重伯努利实验(0-1分布)中，事件A在一次实验中发生的概率是$p_n$，如果当$n\rightarrow\infin$时，有$np_n\rightarrow\lambda$，则
  $$
  \lim_{n\rightarrow\infin}{n\choose k}p_n^k(1-p_n)^{n-k}=\frac{\lambda^k}{k!}e^{-\lambda}
  $$

- 超几何分布

  N件产品里由M件不合格，不放回地抽取n件，其中含有的不合格的件数$X$服从超几何分布

- 几何分布

  每次试验中，事件A发生的概率为p，如果X为事件A首次出现的试验次数，X服从几何分布，记为$X\sim Ge(p)$，分布列为
  $$
  P(X=k)=(1-p)^{k-1}p
  $$

  - 期望

    $E(X)=\frac{1}{p}$

  - 方差

    $Var(X)=\frac{1-p}{p^2}$

- 负二项分布

  也叫帕斯卡分布

### 2.5 常用连续分布

