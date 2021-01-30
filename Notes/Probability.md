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

- k阶矩

  - 随机变量$X$的k阶原点矩
    $$
    \mu_k=E(X^k)
    $$
    故期望就是一阶原点矩

  - 随机变量$X$的k阶中心距
    $$
    v_k=E(X-E(X))^k
    $$
    故方差就是二阶中心距

  - 关系
    $$
    v_k=\sum_{i=1}^k{k\choose i}\mu_i(-\mu_i)^{k-i}
    $$

- 变异系数

  目的：比较两个随机变量的波动大小
  $$
  C_v(X)=\frac{\sqrt{Var(X)}}{E(X)}=\frac{\sigma(X)}{E(X)}
  $$
  无量纲

- 分位数

  目的：求解$F(x)\le p$，解为$\{x\le x_p\}$

  - 下侧p分位数：
    $$
    F(x_p)=\int_{-\infin}^{x_p}p(x)dx=p
    $$
    $x_p$为p分位数

  - 上侧p分位数：
    $$
    1-F(x_p^{'})=\int_{x_p^{'}}^{\infin}p(x)dx=p
    $$
    $x_p^{'}$为p分位数

- 中位数
  $$
  F(x_{0.5})=\int_{-\infin}^{x_{0.5}}p(x)dx=0.5
  $$
  $x_{0.5}$是中位数

- 偏度系数

  目的：描述分布偏离对称性程度
  $$
  \beta_S = \frac{v_3}{v_2^{\frac{3}{2}}} = \frac{E(X-EX)^3}{[E(X-EX)]3}
  $$

- 峰度系数

  目的：描述分布尖峤程度和尾部粗细的特征数
  $$
  \beta_k=\frac{v_4}{v_2^2}=\frac{E(X-EX)^4}{[Var(X)]^2}-3
  $$

# 第三章 多维随机变量及其分布

### 3.1 多维随机变量及其联合分布

- 联合分布函数

  对任一的n个实数$x_1,...x_n$，则n个事件$\{X_1\le x_1\},...,\{X_n\le x_n\}$同时发生的概率
  $$
  F(x_1,...x_n)=P(X_1\le x_1,...,X_n\le x_n)
  $$
  称为n维随机变量$(X_1,...X_n)$的联合分布函数

  - 性质
    1. 单调性：对x，y是单调非减的
    2. 有界性：$0\le F(x,y) \le 1$
    3. 右连续性
    4. 非负性：$P(a<X\le b,c<Y\le d)=F(b,d)-F(a,d)-F(b,c)+F(a,c)\ge 0$

- 联合分布列

  $p_{ij}=P(X=x_i,Y=y_i)$为$(X,Y)$的联合分布列，可用表格形式表示。

  $Sum(p_{ij})=1$

- 联合密度函数
  $$
  F(x,y)=\int_{-\infin}^x\int_{-\infin}^yp(u,v)dvdu
  $$
  称$p(u,v)$为$(X,Y)$的联合密度函数
  $$
  p(x,y)=\frac{\partial^2}{\partial x\partial y}F(x,y)
  $$
  $\int_{-\infin}^\infin\int_{-\infin}^\infin p(x,y)dydx=1$

- 常用多维分布

  - 多项分布

    n次重复试验（有放回），每个结果$A_i$出现的次数为$n_i$，则
    $$
    P(X_1=n_1,...,X_r=n_r)=\frac{n!}{n_1!...n_r!}p_1^{n_1}...p_r^{n_r}
    $$

  - 多维超几何分布

    n次试验（无放回）
    $$
    P(X_1=n_1,...,X_r=n_r)=\frac{({N_1 \choose n_1})...({N_r \choose n_r})}{{N \choose n}}
    $$

  - 多维均匀分布
    $$
    p(x_1,...x_n)=\frac{1}{S_D},D是一个有界区域，S_D是其面积或体积
    $$
    服从D上的多维均匀分布

  - 二元正态分布

    $(X,Y)\sim N(\mu_1,\mu_2,\sigma_1,\sigma_2,\rho)$

    其中$\rho$是X与Y的相关系数

### 3.2 边际分布与随机变量的独立性

- 边际分布函数
  $$
  F_X(x)=\lim_{y\rightarrow \infin}F(x,y)=F(x,\infin)
  $$

- 边际分布列
  $$
  P(X-x_i)=\sum_{j=1}^\infin P(X=x_i,Y=y_i)
  $$

- 边际密度函数
  $$
  p_X(x)=\int_{-\infin}^{\infin}p(x,y)dy
  $$

  1. 多项分布的一维边际分布仍为二项分布
  2. 二维正态分布的边际分布为一维正态分布

- 随机变量间的独立性

  如果$F_i(x_i)$是$X_i$的边际分布函数，有$F(x_1,...x_n)=\prod_{i=1}^nF_i(x_i)$

  则称$X_1,...X_n$相互独立

  - 离散随机变量

    $P(X_1=x_1,...X_n=x_n)=\prod P(X_i=x_i)$

  - 连续随机变量

    $p(x_1,...x_n)=\prod p_i(x_i)$

### 3.3 多维随机变量函数的分布

- 离散随机变量

  - **泊松分布**的可加性

    设$X\sim P(\lambda_1)$,$Y\sim P(\lambda_2)$, $Z=X+Y\sim P(\lambda_1+\lambda_2)$

    即泊松分布的卷积仍为泊松分布，记为$P(\lambda_1)*P(\lambda_2)=P(\lambda_1+\lambda_2)$

  - **二项分布**的可加性

    设$X\sim b(n,p)$,$Y\sim b(m,p)$, $Z=X+Y\sim b(n+m,p)$

    所以服从二项分布$b(n,p)$的随机变量可以分解成n个互相独立的0-1分布的随机变量之和。

  - **最大值**分布

    $Y=\max\{X_1,...X_n\}$

    若$X_i\sim F_i(x)$，则$F_Y(y)=\prod_{i=1}^n F_i(y)$

    若$X_i\sim F(x)$，则$F_Y(y)= [F_i(y)]^n$，$p_Y(y)=F^{'}_Y(y)=n[F_i(y)]^{n-1}p(y)$

  - **最小值**分布

    $Y=\min\{X_1,...X_n\}$

    若$X_i\sim F_i(x)$，则$F_Y(y)=1-\prod_{i=1}^n[1 - F_i(y)]$

    若$X_i\sim F(x)$，则$F_Y(y)= 1-[1-F_i(y)]^n$，$p_Y(y)=F^{'}_Y(y)=n[1-F_i(y)]^{n-1}p(y)$

- 连续场合的卷积公式

  X,Y是两个相互独立的连续随机变量，密度函数分布是$p_X(x),p_Y(y)$,则$Z=X+Y$的密度函数为

  $p_Z(z)=\int_{-\infin}^\infin p_X(z-y)p_Y(y)dy$

  or $p_Z(z)=\int_{-\infin}^\infin p_X(x)p_Y(z-x)dy$

  - **正态分布**的可加性

    两个独立的正态变量之和仍为正态变量

    $N(\mu_1,\sigma_1^2)*N(\mu_2,\sigma_2^2)=N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$

  - **伽马分布**的可加性

    两个尺度参数相同的独立的伽马变量之和仍为伽马变量

    $Ga(\alpha_1,\lambda)*Ga(\alpha_2,\lambda)=Ga(\alpha_1+\alpha_2,\lambda)$

    1. 特例：**指数分布** $Exp(\lambda)=Ga(1,\lambda)$

       m个指数分布之和为伽马变量

    2. 特例：**卡方分布** $\chi^2(n)=Ga(n/2,1/2)$

       m个独立卡方变量之和为卡方变量

       $X_1,...X_n$是n个标准正态变量，其平方和服从自由度为n的卡方分布

- 变量变换法

  - 变量变换法

    如果二维随机变量(X,Y)的联合密度函数为P(x,y)，

    $u=g_1(x,y),\ v=g_2(x,y)$

    $x=x(u,v),\ y=y(u,v)$

    雅可比行列式为：$J=\frac{\partial(x,y)}{\partial(u,v)}=\
    \left|\begin{array}{cccc} 
    \frac{\partial x}{\partial u} &    \frac{\partial y}{\partial u}    \\ 
    \frac{\partial x}{\partial v} &    \frac{\partial y}{\partial v}  \end{array}\right| $

    则(U,V)的联合密度函数为

    $p(u,v)=p(x(u,v),y(u,v))|J|$

  - 增补变量法

    1. 积的公式

       U=XY的密度函数为

       $p_U(u)=\int_{-\infin}^\infin p_X(u/v)p_Y(v)\frac{1}{|v|}dv$

    2. 商的公式

       U=X/Y的密度函数为

       $p_U(u)=\int_{-\infin}^\infin p_X(uv)p_Y(v)|v|dv$

### 3.4 多维随机变量的特征数：协方差/相关系数

- $Z=g(X,Y)$的期望

  - 离散

    联合分布列为$P(X=x_i,Y=y_i)$

    则$E(Z)=\sum_i\sum_j g(x_i,y_j)P(X=x_i,Y=y_i)$

  - 连续

    联合密度函数$p(x,y)$

    则$E(Z)=\int_{-\infin}^\infin\int_{-\infin}^\infin g(x,y)p(x,y)dxdy$

- 期望与方差的运算性质

  1. $E(X+Y)= E(X)+E(Y)$

  2. $E(XY)=E(X)E(Y)$

  3. 若$X$与$Y$不相关，则$Var(X\pm Y)=Var(X)+Var(Y)$

     独立变量代数和的方差等于各方差之和，但对标准差不成立。

     独立随机变量无论相加还是相减，方差只会增加不会减少。

  4. 若$X$与$Y$取任意关系，则$Var(X\pm Y)=Var(X)+Var(Y)\pm 2Cov(X,Y)$

- 协方差(相关中心矩)

  描述两个分量见的相互关联程度。

  $Cov(X,Y)=E[(X-E(X))(Y-E(Y))]$

  $Cov(X,Y)=E(XY)-E(X)E(Y)$

  特别有$Cov(X,X)=Var(X)$

  1. $Cov(X,Y)>0$：正相关
  2. $Cov(X,Y)<0$：负相关
  3. $Cov(X,Y)=0$：不相关，要么毫无关联，要么存在非线性关系。

  不相关与独立的关系：独立一定不相关；不相关不一定独立。

  - 性质1：任意随机变量与常数的协方差为0.
  - 性质2：$Cov(aX,bY)=abCov(X,Y)$
  - 性质3：$Cov(X+Y,Z)=Cov(X,Z)+Cov(Y,Z)$

- 相关系数

  为了相除量纲的影响。

  $(X,Y)$是一个二维随机变量，且$Var(X)=\sigma_X^2>0,Var(Y)=\sigma_Y^2>0$

  则X与Y的线性相关系数：$Corr(X,Y)=\frac{Cov(X,Y)}{\sigma_X \sigma_Y}$

  与协方差关系：是相应标准化变量的协方差

  $Cov(\frac{X-\mu_X}{\sigma_X},\frac{Y-\mu_Y}{\sigma_Y})=Corr(X,Y)$

  - 性质1：$|Corr(X,Y)|\le 1$

  - 性质2：$Corr(X,Y)=\pm 1$的充要条件是

    X与Y之间几乎处处有线性关系，即存在a与b使得$P(Y=aX+b)=1$

  - 性质3：线性相关系数的大小刻画了线性关系的强弱，等于1时完全正相关，等于-1时完全负相关。

- 期望向量与协方差矩阵

  - 期望向量

    n维随机变量为$X=(X_1,...X_n)^{'}$，期望向量为

    $E(X)=(E(X_1),...E(X_N))^{'}$

  - 协方差阵
    $$
    E[(X-E(X)(X-E(X))^{'}]=\left|\begin{array}{ccc} 
    Var(X_1) &  ...  &  Cov(X_1,X_n) \\ 
    ...&...&...\\
    Cov(X_n,X_1)&   ... &Var(X_n) \end{array}\right|
    $$
    性质：n维随机变量的协方差阵是一个对称的非负定矩阵。

  - n元正态分布

    n维随机变量为$X=(X_1,...X_n)^{'}$的协方差阵为$B=Cov(X)$，期望向量为$a=(a_1,...a_n)^{'}$，记$x=(x_1,...x_n)^{'}$，则n元正态分布的密度函数为
    $$
    p(x)=\frac{1}{(2\pi)^{n/2}|B|^{1/2}}e^{-\frac{1}{2}(x-a)^{'}B^{-1}(x-a)}
    $$

### 3.5 条件分布与条件期望

- 条件分布

  - 离散

    - 条件分布列
      $$
      p_{i|j}=P(X=x_i|Y=y_i)=\frac{P(X=x_i,Y=y_i)}{P(Y=y_i)}=\frac{p_{ij}}{p_{*j}}
      $$
      其中$P(Y=y_j)=p_{*j}=\sum_{i=1}^\infin p_{ij}$

    - 条件分布函数
      $$
      F(x|y_j)=\sum_{x_i\le x}P(X=x_i|Y=y_i)=\sum_{x_i\le x}p_{i|j}
      $$

  - 连续

    - 条件分布函数
      $$
      F(x|y)=\int_{-\infin}^\infin \frac{p(u,y)}{p_Y(y)}du
      $$

    - 条件密度函数
      $$
      p(x|y)=\frac{p(x,y)}{p_Y(y)}
      $$
      可改写为$p(x,y)=p_Y(y)p(x|y)$

      二维正态分布的边际分布和条件分布都是一维正态分布.

    - 全概率公式的密度函数
      $$
      p_Y(y)=\int_{-\infin}^\infin p_X(x)p(y|x)dx
      $$

    - 贝叶斯公式的密度函数
      $$
      p(x|y)=\frac{p_X(x)p(y|x)}{\int_{-\infin}^\infin p_X(x)p(y|x)dx}
      $$

- 条件数学期望

  - 二维离散

    $E(X|Y=y)=\sum_ix_iP(X=x_i|Y=y)$

  - 二维连续

    $\int_{-\infin}^\infin xp(x|y)dx$

  - 记法

    - E(X|Y=y)是y的函数,我们可以记$g(y)=E(X|Y=y)$
    - 进一步可记$g(Y)=E(X|Y)$,$E(X|Y)$本身也是随机变量

  - 重期望公式

    $E(X)=E(E(X|Y))$

    - 离散: $E(X)=\sum_jE(X|Y=y_j)P(Y=y_i)$
    - 连续: $E(X)=\int_{-\infin}^\infin E(X|Y=y)p_Y(y)dy$

  - 随机个随机变量和的数学期望

    设$X_1,X_2...$为一列独立同分布的随机变量, 随机变量N只取正整数值, 且N与{$X_n$}独立, 则

    $E(\sum_{i=1}^NX_i)=E(X_1)E(N)$



# 第四章 大数定律与中心极限定理

### 4.1 随机变量序列的收敛性

- 依**概率**收敛: 用于大数定律

  设{$X_n$}为一随机变量序列, X为一随机变量,如果对任一的$\epsilon<0$, 有

  $P(|X_n-X)<\epsilon)\rightarrow 1(n\rightarrow \infin)$

  则称序列{$X_n$}依概率收敛于X, 记作$X_n\rightarrow X$

  含义:绝对偏差$|X_n-X|$小于任一给定量的可能性将随着n增大而越来越接近于1

  当$P(X=c)=1$时, $X_n\rightarrow c$

- 按**分布**收敛: 用于中心极限定理

  设随机变量$X,X_1,...$的分布函数分别为$F(x),F_1(x)...$, 若对F(x)的任一连续点,都有

  $\lim_{n\rightarrow\infin}F_n(x)=F(x)$

  则称{$F_n(x)$}弱收敛于$F(x)$, 记作$F_n(x)\rightarrow F(x)$ or $X_n \rightarrow X$

### 4.2 特征函数

- 定义

  设X是一个随机变量,称$\phi(t)=E(e^{itX}), -\infin<t<\infin$ 为X的特征函数.

  - 离散

    $\phi(t)=\sum_{k=1}^\infin e^{itx_k}p_k$

  - 连续: 是概率密度函数的傅立叶变换

    $\phi(t)=\int_{-\infin}^\infin e^{itx}p(x)dx$

  ![1561426092037](assets/1561426092037.png)

  ![1561426110322](assets/1561426110322.png)

  ![1561426147464](assets/1561426147464.png)

- 性质

  1. $|\phi(t)|\le\phi(0)=1$

  2. $\phi(-t)=\overline{\phi(t)}$

  3. 独立随机变量和的特征函数为每个随机变量的特征函数的积

     $\star$ $\phi_{X+Y}(t)=\phi_X(t)\phi_Y(t)$

  4. 若$E(X^l)$存在,则$\phi(t)$可$l$次求导,对于$1\le k \le l$

     $\phi^{(k)}=i^kE(X^k)$

  5. 一致连续性

     随机变量X的特征函数$\phi(t)$在$(-\infin,\infin)$上一直连续.

  6. 非负定性

     随机变量X的特征函数$\phi(t)$是非负定的, 对任一正整数n及n个实数$t_1,t_2,...$和n个复数$z_1,z_2,...$有

     $\sum_{k=1}^n\sum_{j=1}^n \phi(t_k-t_j)z_k\bar{z_j}\ge 0$

- 常用分布的特征函数

  大约十几种常见分布.

- 特征函数唯一决定分布函数

  - 逆转公式

    $F(x)$是X的分布函数,$\phi(t)$是X的特征函数, 则对$F(x)$的任意两个连续点$x_1<x_2$, 有
    $$
    F(x_2)-F(x_1)=\lim_{T\rightarrow \infin}\frac{1}{2\pi}\int_{-T}^T \frac{e^{-itx_1}-e^{-itx_2}}{it}\phi(t)dt
    $$

  - 唯一性定理(傅里叶逆变换)

    若X为连续随机变量, 如果$\int_{-\infin}infin |\phi(t)|dt<\infin$, 则
    $$
    p(x)=\frac{1}{2\pi}\int_{-\infin}^\infin e^{-itx}\phi(t)dt
    $$

  - 特征函数的连续性定理

    分布函数序列{$F_n(x)$}弱收敛于分布函数F(x)的充要条件是{$F_n(x)$}的特征函数序列{$\phi_n(t)$}收敛于$F(x)$的特征函数$\phi(t)$.

    表明分布函数与特征函数的一一对应关系有连续性

### 4.3 大数定律

讨论的是什么条件下, 随机变量序列的算术平均依概率收敛到其均值的算术平均.

- 伯努利大数定律

  设$S_n$为n重伯努利试验中事件A发生的次数, p为每次试验中A出现的概率, 则对任意的$\epsilon >0$, 有
  $$
  \lim_{n\rightarrow \infin}P(|\frac{S_n}{n}-p|\le \epsilon)=1
  $$
  即频率稳定与概率.

- 大数定律的一般形式

  $\frac{S_n}{n}=\frac{1}{n}\sum_{i=1}^nX_i$,  $p=E(\frac{1}{n}\sum_{i=1}^nX_i)=\frac{1}{n}\sum_{i=1}^n E(X_i)$, 有
  $$
  \lim_{n\rightarrow\infin}P(|\frac{1}{n}\sum_{i=1}^n X_i - \frac{1}{n}\sum_{i=1}^n E(X_i)|<\epsilon)=1
  $$
  则该随机变量服从大数定律

- 切比雪夫大数定律

  假定: 要求互不相关

  设{$X_n$}为一列**两两不相关**的随机变量序列, 且其方差具有共同的上界,即$Var(X_i)\le c$, 则{$X_n$}服从大数定律.

- 马尔科夫大数定律

  假定: 没有任何同分布\独立性\不相关的假定

  - 马尔科夫条件

    $\frac{1}{n^2}Var(\sum_{i=1}^n X_i)\rightarrow 0$

  对随机变量序列{$X_n$}, 若马尔科夫条件成立, 则{$X_n$}服从大数定律.

- 辛钦大数定律

  假定:独立同分布

  设{$X_n$}为一独立分布的随机变量序列, 若$X_i$的数学期望存在, 则{$X_n$}服从大数定律.

  由此得出: 如果{$X_n$}为一独立同分布的随机变量序列,且$E{|X_i|^k}$存在, 其中k是正整数, 则{$X_n^k$}服从大数定律.

### 4.4 中心极限定理

讨论的是什么条件下, 独立随机变量和$Y_n=\sum_{i=1}^nX_i$的分布函数会收敛于正态分布.

- 林德伯格-莱维中心极限定理(Lindeberg-Levy)

  假定: 独立同分布

  设{$X_n$}是独立同分布的随机变量序列, 且$E(X_i)=\mu, Var(X_i)=\sigma^2>0$存在, 若记

  $Y_n^*=\frac{X_1+...X_n-n\mu}{\sigma\sqrt{n}}$, 则对任意实数y, 有
  $$
  \lim_{n\rightarrow\infin}P(Y_n^*\le y)=\Phi(y)=\frac{1}{\sqrt{2\pi}}\int_{-\infin}^ye^{-t^2/2}dt
  $$
  由此得出: 测量误差近似地服从正态分布.

- 棣莫夫-拉普拉斯zhogn中心极限定理(DeMoivre-Laplace)

  专门针对二项分布的正态近似

  设n重伯努利试验中, 事件A在每次试验中出现的概率为p, 记$S_n$为n此试验中事件A出现的次数,且记

  $Y_n^*=\frac{S_n-np}{\sqrt{npq}}$, 则对任意实数y, 有
  $$
  \lim_{n\rightarrow\infin}P(Y_n^*\le y) = \Phi(y)=\frac{1}{2\pi}\int_{-\infin}^ye^{-t^2/2}dt
  $$

- 林德伯格中心极限定理(Lindeberg)

  假定: 独立不同分布

  设独立随机变量序列{$X_n$}满足林德伯格条件, 则对任意的x, 有
$$
  \lim_{n\rightarrow\infin}P(\frac{1}{B_n}\sum_{i=1}^n (X_i-\mu_i)\le x)=\frac{1}{\sqrt{2\pi}}\int_{-\infin}^x e^{-t^2/2}dt
$$

- 李雅普诺夫中心极限定理

  设{$X_n$}为独立随机变量序列,若存在$\delta>0$, 满足

  $\lim_{n\rightarrow\infin}\frac{1}{B^{2+\delta}_n}\sum_{i=1}^nE(|X_i-\mu_i|^{2+\delta})=0$

  则对任意的x, 有
  $$
  \lim_{n\rightarrow\infin}P(\frac{1}{B_n}\sum_{i=1}^n (X_i-\mu_i)\le x)=\frac{1}{\sqrt{2\pi}}\int_{-\infin}^x e^{-t^2/2}dt
  $$



# 第五章 统计量及其分布

### 5.1 总体与样本

### 5.2 样本数据的整理与显示

- 经验分布函数

  $F_n(x)=k/n, 当x_{(k)}\le x < x_{(k+1)}, k=1,2,...,n-1$

- 格里文科定理

  设$x_1,...x_n$是取自总体分布函数为$F(x)$的样本, $F_n(x)$是其经验分布函数,当$n\rightarrow\infin$时, 有
  $$
  P(\sup_{-\infin<x<\infin}|F_n(x)-F(x)|\rightarrow0)=1
  $$
  当n相当大时,经验分布函数是总体分布函数$F(x)$的一个良好的近似.

### 5.3 统计量及其分布 

- 抽样分布

  设$x_1,x_2,...x_n$是来自某个总体的样本, $\bar{x}$是样本均值, 若总体分布为$N(\mu,\sigma^2)$, 则

  $\bar{x}$的精确分布为$N(\mu,\sigma^2/n)$

- 样本方差与样本标准差

  - 样本方差(无偏方差)

    在n不大时, 常用$s^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2$

  - 样本标准差

    $s=\sqrt{s^2}$

  - 偏差平方和

    $\sum_{i=1}^n(x_i-\bar{x})^2=\sum x_i^2-\frac{(\sum x_i)^2}{n}=\sum x_i^2-n\bar{x}^2$

  - 偏差平方的自由度$n-1$

    含义: 在$\bar{x}$确定后, n个偏差中只有n-1个可以自由变动, 而第n个则不能自由取值.

  - 定理

    设总体X具有二阶矩, 即$E(X)=\mu, Var(X)=\sigma^2<\infin$, $x_1,...,x_n$为从总体得到的样本,则

    $E(\bar{x})=\mu, Var(\bar{x})=\sigma^2/n, E(s^2)=\sigma^2$

- 样本矩及其函数

  - k阶原点矩

    $a_k=\frac{1}{n}\sum_{i=1}^n x_i^k$

    样本均值: 样本一阶原点矩

  - k阶中心矩

    $b_k=\frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})^k$

    样本方差: 样本二阶中心矩

  - 样本偏度

    $\hat{\beta_s}=\frac{b_3}{b_2^{3/2}}$

    反映了样本数据与对称性偏离程度和偏离方向

  - 样本峰度

    $\hat{\beta_k}=\frac{b4}{b_2^2}-3$

    $\hat{\beta_k}>>0$: 尖顶型

    $\hat{\beta_k}<<0$: 平顶型

- 次序统计量

  - 单个次序统计量

    总体X的密度函数为p(x), 分布函数为F(x), 则第k个次序统计量$x_{(k)}$的密度函数为

    $p_k(x)=\frac{n!}{(k-1)!(n-k)!}(F(x))^{k-1}(1-F(x))^{n-k}p(x)$

  - 多个次序统计量

    次序统计量$(x_{(i)},x_{(j)})$(i<j)的联合分布密度函数为

    $p_{ij}(y,z)=\frac{n!}{(i-1)!(j-i-1)!(n-j)!}[F(y)^{i-1}][F(z)-F(y)]^{j-i-1}[1-F(z)]^{n-j}p(y)p(z)$

  - 样本极差

    $R_n=x_{(n)}-x_{(1)}$

- 样本分位数与中位数

  - 中位数
    $$
    m_{0.5}=\begin{equation}  
    \left\{  
                 \begin{array}{**lr**}  
                 x_{(\frac{n+1}{2})} & n为奇数\\  
                 \frac{1}{2}(x_{(\frac{n}{2})}+x_{(\frac{n}{2}+1)})& n为偶数\\  
                 \end{array}  
    \right.  
    \end{equation}  
    $$

  - p分位数
    $$
    m_{p}=\begin{equation}  
    \left\{  
                 \begin{array}{**lr**}  
                 x_{[np+1])} & np不是整数\\  
                 \frac{1}{2}(x_{(np)}+x_{(np+1)})& np是整数\\  
                 \end{array}  
    \right.  
    \end{equation}  
    $$
    当$n\rightarrow\infin$时, p分位数的渐进分布是
    $$
    m_p\sim N(x_p,\frac{p(1-p)}{np^2(x_p)})
    $$

### 5.4 三大抽样分布

[*三大抽样分布*:卡方分布,t分布和F分布的简单理解](https://www.baidu.com/link?url=57aywD0Q6WTnl7XKbIHuE7lcWGXh50Vy3z1lItKlmdAUkrVVxJo8WsaylEN5xRxQbxqY2uHxWIYp7UW2oc9hYg26z3yxFDDyFxMdMVmifjC&wd=&eqid=dbe5cd2a000f3278000000035d130bdd)

- 卡方分布

  设$X_1,...X_n$独立同分布与标准正态分布$N(0,1)$, 则$\chi^2=X_1^2+...+X_n^2$的分布称为自由度为n的$\chi^2$分布, 记为$\chi^2\sim\chi^2(n)$

- F分布

  设随机变量$X1\sim\chi^2(m), X_2\sim\chi^2(m)$, $X_1$与$X_2$独立, 则称$F=\frac{X_1/m}{X_2/m}$的分布是自由度为m与n的F分布, 记为$F\sim F(m,n)$, 其中m为分子自由度, n为分母自由度.

- t分布

  设随机变量$X_1$与$X_2$独立且$X_1\sim N(0,1),X_2\sim \chi^2(n)$, 则称$t=\frac{X_1}{\sqrt{X_2/n}}$的分布为自由度为n的t分布, 记为$t\sim t(n)$

![1561529662310](assets/1561529662310.png)

### 5.5 充分统计量

- 定义

  设$x_1,...x_n$是来自某个总体的样本, 总体分布函数为$F(x;\theta)$, 统计量$T=T(x_1,...,x_n)$称为$\theta$的充分统计量.

  如果在给定T的取值后, $x_1,...x_n$的条件分布与$\theta$无关.

- 因子分解定理

  设总体概率函数为$F(x;\theta)$, $x_1,...x_n$是样本,  则$T=T(x_1,...,X_n)$为充分统计量的充要条件是:

  存在两个函数$g(t,\theta)$和$h(x_1,...x_n)$使得对任意的$\theta$和任一组观测值$x_1,...x_n$, 有

  $f(x_1,...x_n;\theta)=g(T(x_1,...x_n),\theta)h(x_1,...,x_n)$

  其中$g(t,\theta)$是通过统计量T的取值而依赖于样本的.



# 第六章 参数估计

### 6.1 点估计的概念与无偏性

- 无偏性

  设$\hat{\theta}=\hat{\theta}(x_1,...,x_n)$是$\theta$的一个估计, $\theta$的参数空间为$\Theta$, 若对任意的$\theta\in\Theta$, 有

  $E_{\theta}(\hat{\theta})=theta$, 则称$\hat{\theta}$是$\theta$的无偏估计.

- 有效性

  设$\hat{\theta_1}$,$\hat{\theta_2}$是$\theta$的两个无偏估计, 如果对任意的$\theta\in\Theta$有

  $Var(\hat{\theta_1})\le Var(\hat{\theta_2})$, 且至少有一个$\theta\in\Theta$使得上述不等号严格成立,

  则称$\hat{\theta_1}$比$\hat{\theta_2}$有效.

### 6.2 矩估计及相合性

- 矩估计

  用样本矩去替换总体矩, 用经验分布函数去替换总体分布.

  其理论基础是格里文科定理.

- 相合性

  随着样本量的不断增大, 经验分布函数逼近真实分布函数, 因此完全可以要求估计量随着样本量的不断增大而逼近参数真值.

  - 定义

    设$\theta\in\Theta$为未知参数, $\hat{\theta_n}=\hat{\theta_n}(x_1,...x_n)$是$\theta$的一个估计量, n是样本容量, 若对任何一个$\epsilon>0$, 有

    $\lim_{n\rightarrow\infin}P(|\hat{\theta_n}-\theta|\ge\epsilon)=0$

    则称$\hat{\theta_n}$为参数$\theta$的相合估计.

  - 定理

    设 $\hat{\theta_n}=\hat{\theta_n}(x_1,...x_n)$是$\theta$的一个估计量, 若$lim_{n\rightarrow\infin}E(\hat{\theta_n})=\theta$, $\lim_{n\rightarrow\infin}Var(\hat{\theta_n})=0$,

    则$\hat{\theta_n}$是$\theta$的相合估计

### 6.3 最大似然估计与EM算法

- 最大似然估计

  - 似然函数$L(\theta)$

    设总体的概率函数为$p(x;\theta)$, $\theta\in \Theta$, 其中$\theta$是一个未知参数组成的参数向量, $x_1,...x_n$是来自该总体的样本, 将样本的联合概率密度看成$\theta$的函数, 称为样本的似然函数

    $L(\theta)=L(\theta;x_1,x_2,...,x_n)=p(x_1;\theta)p(x_2;\theta)...p(x_n;\theta)$

    也可定义为$\ln L(\theta)$

  - 最大似然估计$MLE$$(maximum likelihood estimate)$

    如果某统计量$\hat{\theta}=\hat{\theta}(x_1,...x_n)$满足$L(\hat{\theta})=\max_{\theta\in\Theta}L(\theta)$

    则称$\hat{\theta}$是$\theta$的最大似然估计

    - 最大似然估计的不变性

      如果$\hat{\theta}$是$\theta$的最大似然估计,则对任意一函数$g(\theta)$, 其最大似然估计为$g(\hat{\theta})$.

- EM算法

  - 思路

    Expectation: 第一步求期望,以便把多余的部分去掉

    Maximization: 第二步求极大值得到MLE

  - 算法

    ![1562505923097](assets/1562505923097.png)

### 6.4 最小方差无偏估计

- 均方误差
