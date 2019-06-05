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

- 正态分布

  $X \sim N(\mu,\sigma^2)$

  - 密度函数
    $$
    p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

  - 分布函数
    $$
    F(x)=\frac{1}{\sqrt{2\pi}\sigma}\int_{-\infin}^xe^{-\frac{(x-\mu)^2}{2\sigma^2}}dt
    $$

  - 期望

    $E(X)=\mu$

  - 方差

    $Var(X)=\sigma^2$

- 标准正态分布

  此时$\mu=0$, $\sigma=1$
  $$
  \phi(x) =\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2\sigma^2}} \\
  \Phi(x) =\frac{1}{\sqrt{2\pi}}\int_{-\infin}^xe^{-\frac{(x)^2}{2\sigma^2}}dt
  $$

- 正态变量的标准化

  定理：若随机变量$X\sim N(\mu,\sigma^2)$，

  则$U=\frac{X-\mu}{\sigma}\sim N(0,1)$

  则$P(X\le c)=\Phi(\frac{c-\mu}{\sigma})$

  $X$的标准化随机变量 $X^*=\frac{X-\mu}{\sigma}$，且$E(X^*)=0$ $Var(X^*)=1$

- 均匀分布

  $X\sim U(a,b)$

  - 密度函数
    $$
    p(x) = \frac{1}{b-a},a<x<b
    $$

  - 分布函数
    $$
    F(x)=0,x<a\\
    F(x)=\frac{x-a}{b-a},a\le x<b\\
    F(x) = 1,x\ge b
    $$

  - 期望

    $E(X)=\frac{a+b}{2}$

  - 方差

    $Var(X)=\frac{(b-a)^2}{12}$

- 指数分布

  $X \sim Exp(\lambda)$

  - 密度函数
    $$
    p(x)=\lambda e^{-\lambda x},x\ge 0
    $$

  - 分布函数
    $$
    F(x)=1-e^{-\lambda x},x\ge 0
    $$

  - 期望

    $E(X)=\frac{1}{\lambda}$

  - 方差

    $Var(X)=\frac{1}{\lambda^2}$

  与泊松分布的关系：如果某设备在长为t的时间$[0,t]$内发生故障的次数N(t)服从参数为$\lambda t$的泊松分布，则相继两次故障之间的时间间隔T服从参数为$\lambda$的指数分布。

- 伽马分布

  $X \sim Ga(\alpha,\lambda)$

  - 伽马函数
    $$
    \Gamma(\alpha)=\int_0^\infin x^{\alpha-1}e^{-x}dx
    $$

  - 密度函数
    $$
    p(x)=\frac{\lambda^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x}
    $$

  - 期望

    $E(x)=\frac{\alpha}{\lambda}$

  - 方差

    $Var(X)=\frac{\alpha}{\lambda^2}$

  - 特例

    - 指数分布

      当$\alpha=1$时

    - 卡方分布

      当$\alpha=n/2,\lambda=1/2$，n是自然数。此时是自由度为n的卡方分布
      $$
      Ga(\frac{n}{2},\frac{1}{2})=\chi^2(n)
      $$
      卡方分布期望是$n$，方差是$2n$

- 贝塔分布

  $X \sim Be(a,b)$

  - 贝塔函数
    $$
    B(a,b)=\int_0^1x^{a-1}(1-x)^{b-1}dx
    $$
    性质：$B(a,b)=B(b,a)$

    与伽马函数关系：$B(a,b)=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$

  - 密度函数
    $$
    p(x)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1},0<x<1
    $$

  - 期望

    $E(X)=\frac{a}{a+b}$

  - 方差

    $Var(X)=\frac{ab}{(a+b)^2(a+b+1)}$

### 2.6 随机变量函数的分布

问题：$y=g(x)$，已知$X$的分布，求出$Y$的分布

- 离散随机变量函数的分布

  计算分布列，再合并就可以

- 连续随机变量函数的分布

  - $g(x)$严格单调

    定理1：设$X$是连续随机变量，其密度函数为$p_X(x)$. $Y=g(X)$是另一个随机变量

    若$y=g(x)$严格单调，其反函数$h(y)$有连续导函数，则$Y=g(X)$的密度函数为
    $$
    p_Y(y)=p_X[h(y)]|h^{'}(y)|,a<y<b
    $$

    - $X$服从正态分布$N(\mu,\sigma^2)$，则$a\neq 0$时有
      $$
      Y=aX+b \sim N(a\mu +b,a^2\sigma^2)
      $$

    - 对数正态分布：X服从正态分布$N(\mu,\sigma^2)$，则
      $$
      Y=e^{X}\sim LN(\mu,\sigma^2)
      $$

    - $X$服从伽马分布$Ga(\alpha,\lambda)$，则当$k>0$时
      $$
      Y=kX \sim Ga(\alpha,\lambda/k)
      $$
      可将任一伽马分布转换为卡方分布

    定理2：设$X$的分布函数$F_X(x)$为严格单调增的连续函数，其反函数$F_X^{-1}(y)$存在，则

    $Y=F_X(X)$服从$(0,1)$上的均匀分布$U(0,1)$

  - $g(x)$为其他形式

    具体问题具体考虑

### 2.7 分布的其他特征数