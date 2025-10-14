## Important processes

这里记录一些常见的随机过程，特别是WSS随机过程

### 正弦随机过程

正弦随机过程有很多中形式，我们主要以复正弦函数为主，然后提一下实正弦函数

#### 复正弦1

先来关注复正弦
$$
x[n] = A \exp \left(\mathrm{j} (2 \pi f_0 n + \phi) \right)
$$
当$A$和$\phi$都为常数时，$x[n]$不是WSS，因为均值随着$n$在变化。

当$A$是常数，$\phi$为在$[0, 2\pi)$的均匀分布时，$x[n]$是WSS随机过程，可以计算得到其均值和相关分别为
$$
E\left[x[n]\right] = 0
$$

$$
\begin{aligned}
r_{xx}[k] &= E\left[ x[n] x^{*}[n - k] \right] = A^2 \exp(\mathrm{j} 2\pi f_0 k)
\end{aligned}
$$

当$A$是Rayleigh分布时（该分布的均值和方差看Kay书），若均值$E[A] = \sqrt{\pi / 2} A_0$，先考虑单个随机变量
$$
A \exp(\mathrm{j} \phi) \sim \mathcal{CN}(0, A_0^2)
$$
由此，乘一个常数，方差变成常熟模平方倍
$$
x[n] = A \exp(\mathrm{j} \phi) \exp(\mathrm{j} 2\pi f_0 n) \sim \mathcal{CN}(0, A_0^2)
$$
此时，自相关函数为
$$
\begin{aligned}
r_{xx}[k] &= E\left[ x[n] x^{*}[n - k] \right] \\
&= E[A^2] \exp(\mathrm{j} 2\pi f_0 k) \\
&= 2 A_0^2 \exp(\mathrm{j} 2\pi f_0 k)
\end{aligned}
$$


####  实正弦1

根据circular symmetric随机变量和实随机变量的关系，即
$$
\tilde{x} = \bar{x} + \mathrm{j} \tilde{x} \sim \mathcal{CN}(0, \sigma^2)
$$
此时有
$$
\bar{x} \sim \mathcal{N}(0, \frac{\sigma^2}{2}) \\
\tilde{x} \sim \mathcal{N}(0, \frac{\sigma^2}{2})
$$
那么
$$
x[n] = A \cos(2 \pi f_0 n + \phi)
$$
在$A$常数，$\phi$均匀分布的情况下
$$
E\left[x[n]\right] = 0
$$

$$
r_{xx}[k] = \frac{A^2}{2} \cos( 2\pi f_0 k)
$$

当$A$是Rayleigh分布时，若均值$E[A] = \sqrt{\pi / 2} A_0$，则
$$
x[n] \sim \mathcal{N}(0, \frac{A_0^2}{2})
$$

此时的自相关函数为
$$
\begin{aligned}
r_{xx}[k] &= A_0^2 \cos( 2\pi f_0 k)
\end{aligned}
$$


### Z变换两个重要性质

#### 自相关的Z变换

$$
h(-n) \leftrightarrow H^{*}(1 / z^{*})
$$

由此自相关的表达式
$$
y(n) = x(n) * h(-n) = \sum\limits_{k = -\infty}^{\infty} x(k) h(k-n)
$$
对应的Z变换为
$$
Y(z) = X(z) H^*(1 /  z^*)
$$


### 圆对称零点性质

若$A(z)$的零点为$z_i$，那么$1 / z^*_i$是$A^*(1 / z^*)$的零点。

证明
$$
A^*(1 / z^*) \Big|_{z = 1 / z^*_i} = A^*(\frac{1}{ (1 / z^*_i)^*}) = A^*(z_i) = 0
$$



### 波形匹配问题

现在有$K$个能量不一致的已知波形$\mathbf{s}_k$，也有$K$个能量不一致的由每个已知波形未知复幅度增益加噪声的观测$\mathbf{x}_k$，但是这两组信号没有对应上，改用怎样的方法进行对应呢
$$
p_{ij} = \frac{\left| \mathbf{s}_i^H \mathbf{x}_k \right|^2}{\mathbf{s}_i^H \mathbf{s}_i \mathbf{x}_k^H \mathbf{x}_k}
$$
对于每一个$i = 1 \dots K$，找到最大的$j$，就完成了匹配。

简单理解，你可以认为每一个观测就是由$\mathbf{x} = \alpha \mathbf{s}$，所以直接将这个关系代入上式，可以发现刚刚好完成匹配时，值为1。

同时，也可以从Cauchy-Schwarz不等式来理解
$$
\left| \mathbf{s}^H \mathbf{x} \right|^2 \leq \left( \mathbf{s}^H \mathbf{s} \right) \left( \mathbf{x}^H \mathbf{x} \right)
$$
只有当两个向量平行的时候才能成立，也就是上式值最大的时候，刚好匹配。

### 不等式总结

三角不等式（triangle inequality）是关于向量范数的不等式
$$
\left| \| \mathbf{x} \| - \| \mathbf{y} \| \right| \le \| \mathbf{x} + \mathbf{y} \| \le \| \mathbf{x} \| + \| \mathbf{y} \|
$$


Cauchy-Schwarz不等式是关于内积的不等式
$$
| \mathbf{x}^T \mathbf{y} |^2 \leq (\mathbf{x}^T \mathbf{x}) (\mathbf{y}^T \mathbf{y})
$$

### EXIP

参考：On reparametrization of loss functions used in estimation and the invariance principle

对于某一个估计模型，你要估计参数$\boldsymbol{\theta}$，需要最大化似然函数
$$
\hat{\boldsymbol{\theta}} = \arg \max L_{\boldsymbol{\theta}}(\boldsymbol{\theta})
$$
若直接优化$L_{\boldsymbol{\theta}}(\boldsymbol{\theta})$比较复杂，你可以先找到一个中间变量$\boldsymbol{\alpha}(\boldsymbol{\theta})$，使其为$\boldsymbol{\theta}$的函数，由此，你可以优化关于$\boldsymbol{\alpha}$的似然函数
$$
\hat{\boldsymbol{\alpha}} = \arg \max L_{\boldsymbol{\alpha}}(\boldsymbol{\alpha})
$$
在得到$\hat{\boldsymbol{\alpha}}$后，将其视为观测数据，得到一个新的模型
$$
\hat{\boldsymbol{\alpha}} = \boldsymbol{\alpha}(\boldsymbol{\theta}) + \Delta \boldsymbol{\alpha}
$$
其中，$\Delta \boldsymbol{\alpha}$为误差项，其拥有协方差$L_{\boldsymbol{\alpha}}''(\boldsymbol{\alpha})$为似然函数的二阶导，也就是Fisher信息矩阵（Hessian矩阵）。故在第二步中，使用最大似然函数为
$$
\left[ \hat{\boldsymbol{\alpha}} -  \boldsymbol{\alpha}(\boldsymbol{\theta}) \right]^H L_{\boldsymbol{\alpha}}''(\boldsymbol{\alpha}) \left[ \hat{\boldsymbol{\alpha}} -  \boldsymbol{\alpha}(\boldsymbol{\theta}) \right]
$$
理论上证明，在满足特定条件的情况下，这是渐近有效的。但是这个特定条件不怎么理解，需要用的时候再去查文献吧，总的来说，还是没理解这种分步内涵，若理解了，对那个分步估计的工作也是有好处的。

$$

$$



































