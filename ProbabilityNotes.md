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