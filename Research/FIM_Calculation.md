## FIM计算方法

由于计算FIM的方法众多，所以在这里进行一个总结与分类，并提供相应的例子来巩固。

### FIM的定义

最原始FIM定义如下
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = -E\left[ \frac{\partial^2 \ln p(\mathbf{x}; \boldsymbol{\theta})}{\partial \theta_i \partial \theta_j} \right]
$$

### 一般高斯情况的FIM

若接收数据矢量为高斯矢量
$$
\mathbf{x} \sim \mathcal{N} \left( \mathbf{u}(\boldsymbol{\theta}), \mathbf{C}(\boldsymbol{\theta}) \right)
$$
则FIM为
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right]^T \mathbf{C}^{-1}(\boldsymbol{\theta}) \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] + \frac{1}{2} \operatorname{tr}\left[ \mathbf{C}^{-1}(\boldsymbol{\theta}) \frac{\partial \mathbf{C}(\boldsymbol{\theta})}{\partial \theta_i} \mathbf{C}^{-1}(\boldsymbol{\theta}) \frac{\partial \mathbf{C}(\boldsymbol{\theta})}{\partial \theta_j} \right]
$$
特别的，当$\mathbf{u}(\boldsymbol{\theta}_1)$与$\mathbf{C}(\boldsymbol{\theta}_2)$补充和时，此时FIM拥有分块对角的性质。

特别的情况，在**白噪声**的信号的**标量参数**估计问题中，拥有特殊形式
$$
x[n] \sim \mathcal{N}\left( s[n; \theta], \sigma^2 \right)
$$
则有标量FIM为
$$
I(\theta) = \sum\limits_{n = 0}^{N - 1} \left( \frac{\partial s[n; \theta]}{\partial \theta} \right)^2
$$
这个特别例子是已知的确知信号中时延参数的检测Kay书中P39。

### 零均值WSS高斯随机过程的渐近FIM

若接收数据矢量不仅为高斯矢量，还是零均值WSS矢量
$$
\mathbf{x} \sim \mathcal{N} \left( \mathbf{0}, \mathbf{C}(\boldsymbol{\theta}) \right)
$$
其中，$\mathbf{C}(\boldsymbol{\theta})$是拥有Toeplitz结构的协方差矩阵，等效于拥有功率谱密度$P_{xx}(f; \boldsymbol{\theta})$，则其似然函数拥有渐近形式
$$
\ln p(\mathbf{x}) -\frac{N}{2} \ln 2 \pi - \frac{N}{2} \int_{-1 / 2}^{1 / 2} \left( \ln P_{xx}(f; \boldsymbol{\theta}) +  \frac{I(f)}{P_{xx}(f; \boldsymbol{\theta})} \right) \mathrm{d} f \\
I(f) = \frac{1}{N} \left|\sum\limits_{n = 0}^{N} x[n] \exp(\mathrm{j} 2 \pi f n)\right|^2
$$
其中，$I(f)$为数据$\mathbf{x}$的周期图。

则其拥有渐近FIM
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = \frac{N}{2} \int_{-1 / 2}^{1 / 2} \frac{\partial \ln P_{xx}(f; \boldsymbol{\theta})}{\partial \theta_i} \frac{\partial \ln P_{xx}(f; \boldsymbol{\theta})}{\partial \theta_j} \mathrm{d} f
$$
该结果是使用Toeplitz协方差的特征值是傅里叶基这一结论进行推导的，具体参考Kay书P409。

使用WSS高斯随机过程推导的渐近FIM很容易拓展到连续的版本，即接收数据从离散时间序列$x[n]$变成了连续时间信号$x(t)$
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = \frac{T}{2} \int_{-\infty}^{\infty} \frac{\partial \ln P_{xx}(f; \boldsymbol{\theta})}{\partial \theta_i} \frac{\partial \ln P_{xx}(f; \boldsymbol{\theta})}{\partial \theta_j} \mathrm{d} f
$$
其中，$T$代表观测数据的时间窗长度。

### 一般WSS高斯随机过程的渐近FIM

论文：Frequency domain Cramer-Rao bound for Gaussian processes

当接收数据均值不为零的WSS矢量
$$
\mathbf{x} \sim \mathcal{N} \left( \mathbf{u}(\boldsymbol{\theta}), \mathbf{C}(\boldsymbol{\theta}) \right)
$$
其中，$\mathbf{C}(\boldsymbol{\theta})$是拥有Toeplitz结构的协方差矩阵，等效于拥有功率谱密度$P_{nn}(f; \boldsymbol{\theta})$。

同时，这里的WSS矢量是与一般WSS矢量不同，这里不要求均值恒定，换句话说，可以认为接收数据包含一个确定性信号和零均值WSS随机过程
$$
\mathbf{x} = \mathbf{s} + \mathbf{n}
$$
其中，$\mathbf{s}$为确定性信号，$\mathbf{n}$为零均值WSS随机过程。则其FIM可写为
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = N\int_{- 1 /2}^{1 / 2} \frac{{d}_i^*(f;  \boldsymbol{\theta}) {d}_i(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} {d} f + \frac{N}{2} \int_{-1 / 2}^{1 / 2} \frac{\partial \ln P_{nn}(f; \boldsymbol{\theta})}{\partial \theta_i} \frac{\partial \ln P_{nn}(f; \boldsymbol{\theta})}{\partial \theta_j} \mathrm{d} f \\
$$
$$
d_i(f_k; \boldsymbol{\theta}) = \frac{1}{\sqrt{N}} \cdot \operatorname{DTFT}_{k}\left\{ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i}  \right\} \\
$$

同样拓展到连续时间
$$
\left[\mathbf{I}(\boldsymbol{\theta})\right]_{ij} = T\int_{- \infty}^{\infty} \frac{{d}_i^*(f;  \boldsymbol{\theta}) {d}_j(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} \mathrm{d} f + \frac{T}{2} \int_{-\infty}^{\infty} \frac{\partial \ln P_{nn}(f; \boldsymbol{\theta})}{\partial \theta_i} \frac{\partial \ln P_{nn}(f; \boldsymbol{\theta})}{\partial \theta_j} \mathrm{d} f
$$
$$
\begin{aligned}
{d}_i(f; \boldsymbol{\theta}) &= \frac{1}{\sqrt{T}} \operatorname{FT}\left\{ \frac{\partial u(t; \boldsymbol{\theta})}{\partial \theta_i } \right\} \\
&= \frac{1}{\sqrt{T}}  \int_{-\infty}^{\infty} \frac{\partial u(t; \boldsymbol{\theta})}{\partial \theta_i } \exp(-\mathrm{j} 2 \pi f t) \mathrm{d} t
\end{aligned}
$$
其中，这里的${d}_i(f; \boldsymbol{\theta})$为归一后的傅里叶变换后

**这里给出以Kay书中特征值方法的证明：**

零均值与非零均值的不同在于，一般高斯FIM中处理均值的那一部分
$$
\left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right]^T \mathbf{C}^{-1}(\boldsymbol{\theta}) \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right]
$$
具有Toeplitz矩阵结构的协方差矩阵$\mathbf{C}(\boldsymbol{\theta})$拥有近似特征向量和特征值
$$
\mathbf{v}_i = \frac{1}{\sqrt{N}} \begin{bmatrix}
1 & \exp(\mathrm{j} 2 \pi f_i ) & \cdots &\exp\left(\mathrm{j} 2 \pi f_i (N - 1)\right)
\end{bmatrix}^T \\
\lambda_i = P_{nn}(f_i; \boldsymbol{\theta})
$$
则将$\mathbf{C}(\boldsymbol{\theta})$秩1矩阵展开
$$
\mathbf{C}(\boldsymbol{\theta}) = \sum\limits_{k = 0}^{N - 1} \lambda_k \mathbf{v}_k \mathbf{v}_k^H
$$
由此
$$
\begin{aligned}
& \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right]^T \mathbf{C}^{-1}(\boldsymbol{\theta}) \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] \\
&= \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right]^T 
\left(\sum\limits_{k = 0}^{N - 1} \frac{1}{\lambda_k} \mathbf{v}_k \mathbf{v}_k^H\right)
\left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] \\
&= \sum\limits_{k = 0}^{N - 1} \frac{1}{\lambda_k} \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right]^T 
\left(  \mathbf{v}_k \mathbf{v}_k^H\right)
\left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] \\
&= \sum\limits_{k = 0}^{N - 1} \frac{1}{\lambda_k} \left( \mathbf{v}_k^H \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_k} \right] \right)^*
 \left( \mathbf{v}_k^H 
\left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] \right) 
\end{aligned}
$$
由于特征向量$\mathbf{v}_i$是傅里叶基，所以
$$
\begin{aligned}
\mathbf{v}_k^H \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i} \right] &= \frac{1}{\sqrt{N}} \cdot \operatorname{DTFT}_{k}\left\{ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_i}  \right\} \\
&= d_i(f_k; \boldsymbol{\theta})
\end{aligned}
$$
表示对求导后的均值序列在$f_k$处做DTFT变换后使用$\sqrt{N}$进行归一化的标量。

由此
$$
\begin{aligned}
&\sum\limits_{k = 0}^{N - 1} \frac{1}{\lambda_k} \left( \mathbf{v}_k^H \left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_k} \right] \right)^*
 \left( \mathbf{v}_k^H 
\left[ \frac{\partial \mathbf{u}(\boldsymbol{\theta})}{\partial \theta_j} \right] \right) \\
&= N\sum\limits_{k = 0}^{N - 1} \frac{{d}_i^*(f_k;  \boldsymbol{\theta}) {d}_j(f_k; \boldsymbol{\theta})}{P_{nn}(f_k; \boldsymbol{\theta})} \frac{1}{N} \\
&= N\int_{- 1 /2}^{1 / 2} \frac{{d}_i^*(f;  \boldsymbol{\theta}) {d}_j(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} \mathrm{d} f
\end{aligned}
$$



### 复数情况的FIM

这里将上述所有公式的复数情况全部重写一遍

**一般复高斯情况的FIM：**
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = 2 \operatorname{Re} \left\{\left[ \frac{\partial \mathbf{u}(\boldsymbol{\xi})}{\partial \xi_i} \right]^H\mathbf{C}^{-1}(\boldsymbol{\xi}) \left[ \frac{\partial \mathbf{u}(\boldsymbol{\xi})}{\partial \xi_j} \right] \right\}  + \operatorname{tr}\left[ \mathbf{C}^{-1}(\boldsymbol{\xi}) \frac{\partial \mathbf{C}(\boldsymbol{\xi})}{\partial \xi_i} \mathbf{C}^{-1}(\boldsymbol{\xi}) \frac{\partial \mathbf{C}(\boldsymbol{\xi})}{\partial \xi_j} \right]
$$
**零均值WSS高斯随机过程的渐近似然函数：**
$$
\ln p(\mathbf{x}) = -{N} \ln \pi - {N} \int_{-1 / 2}^{1 / 2} \left( \ln P_{xx}(f; \boldsymbol{\xi}) +  \frac{I(f)}{P_{xx}(f; \boldsymbol{\xi})} \right) \mathrm{d} f \\
I(f) = \frac{1}{N} \left|\sum\limits_{n = 0}^{N} x[n] \exp(\mathrm{j} 2 \pi f n)\right|^2
$$
**零均值WSS高斯随机过程的渐近FIM：**
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = {N} \int_{-1 / 2}^{1 / 2} \frac{\partial \ln P_{xx}(f; \boldsymbol{\xi})}{\partial \xi_i} \frac{\partial \ln P_{xx}(f; \boldsymbol{\xi})}{\partial \xi_j} \mathrm{d} f
$$
**零均值WSS高斯随机过程的渐近FIM（连续）：**
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = {T} \int_{-\infty}^{\infty} \frac{\partial \ln P_{xx}(f; \boldsymbol{\xi})}{\partial \xi_i} \frac{\partial \ln P_{xx}(f; \boldsymbol{\xi})}{\partial \xi_j} \mathrm{d} f
$$
**一般WSS高斯随机过程的渐近FIM：**
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = 2 \operatorname{Re}\left\{N\int_{- 1 /2}^{1 / 2} \frac{{d}_i^*(f;  \boldsymbol{\xi}) {d}_i(f; \boldsymbol{\xi})}{P_{nn}(f; \boldsymbol{\xi})} {d} f \right\}+ {N} \int_{-1 / 2}^{1 / 2} \frac{\partial \ln P_{nn}(f; \boldsymbol{\xi})}{\partial \xi_i} \frac{\partial \ln P_{nn}(f; \boldsymbol{\xi})}{\partial \xi_j} \mathrm{d} f \\
$$

$$
d_i(f_k; \boldsymbol{\xi}) = \frac{1}{\sqrt{N}} \cdot \operatorname{DTFT}_{k}\left\{ \frac{\partial \mathbf{u}(\boldsymbol{\xi})}{\partial \xi_i}  \right\} \\
$$

**一般WSS高斯随机过程的渐近FIM（连续）：**
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = 2 \operatorname{Re}\left\{ T\int_{- \infty}^{\infty} \frac{{d}_i^*(f;  \boldsymbol{\xi}) {d}_j(f; \boldsymbol{\xi})}{P_{nn}(f; \boldsymbol{\xi})} \mathrm{d} f \right\} + {T} \int_{-\infty}^{\infty} \frac{\partial \ln P_{nn}(f; \boldsymbol{\xi})}{\partial \xi_i} \frac{\partial \ln P_{nn}(f; \boldsymbol{\xi})}{\partial \xi_j} \mathrm{d} f
$$

$$
\begin{aligned}
{d}_i(f; \boldsymbol{\xi}) &= \frac{1}{\sqrt{T}} \operatorname{FT}\left\{ \frac{\partial u(t; \boldsymbol{\xi})}{\partial \xi_i } \right\} \\
&= \frac{1}{\sqrt{T}}  \int_{-\infty}^{\infty} \frac{\partial u(t; \boldsymbol{\xi})}{\partial \xi_i } \exp(-\mathrm{j} 2 \pi f t) \mathrm{d} t
\end{aligned}
$$

其中，这里的${d}_i(f; \boldsymbol{\xi})$为归一后的傅里叶变换后

### 多测量通道下的FIM

论文：On the Cramer-Rao Bound for Time Delay and  Doppler Estimation

如果每一个快拍中，接收到不止一个数据，而是一个向量，比如DOA问题中一个快拍是$M$个数据，TDOA/FDOA问题中一个快拍是来自两个传感器的数据。最后得到的数据是以矩阵的形式呈现，参数蕴藏在多个测量通道中，该如何拓展前面所述的单通道FIM计算问题？

对于定义和一般高斯的情况，由于同一个快拍接收到的数据都是高斯随机变量，所以直接将最后接收到的矩阵以某种方式排列成向量，并算出相应的均值向量和协方差矩阵，按照上述公式进行即可。例子详见DOA估计的笔记。

假设有两个WSS高斯实随机序列，且两个随机序列的互相关也平稳
$$
\mathbf{x}_1  \sim\mathcal{CN}\left(0, \mathbf{C}_1(\boldsymbol\xi)\right)  \\ 

\mathbf{x}_2  \sim\mathcal{CN}\left(0, \mathbf{C}_2(\boldsymbol\xi)\right)
$$
分别对应功率谱密度为$P_{x_1x_1}(f; \boldsymbol\xi)$和$P_{x_2x_2}(f; \boldsymbol\xi)$，以及互功率谱$P_{x_1 x_2}(f; \boldsymbol\xi)$，将功率谱密度写为矩阵形式
$$
\mathbf{P}(f; \boldsymbol\xi) = \begin{bmatrix}
P_{x_1x_1}(f; \boldsymbol\xi) & P_{x_1 x_2}(f; \boldsymbol\xi) \\
P_{x_2 x_1}(f; \boldsymbol\xi) & P_{x_2x_2}(f; \boldsymbol\xi)
\end{bmatrix}
$$
则可计算FIM（实信号）
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = \frac{N}{2} \int_{- 1 /2}^{1 / 2} \operatorname{tr}\left\{ \mathbf{P}^{-1}(f; \boldsymbol\xi) \frac{\partial \mathbf{P}(f; \boldsymbol\xi)}{\partial\xi_i}\mathbf{P}^{-1}(f; \boldsymbol\xi) \frac{\partial \mathbf{P}(f; \boldsymbol\xi)}{\partial\xi_j} \right\} \mathrm{d} f
$$
相应的连续情况为（实信号）
$$
\left[\mathbf{I}(\boldsymbol{\xi})\right]_{ij} = \frac{T}{2} \int_{- \infty}^{\infty} \operatorname{tr}\left\{ \mathbf{P}^{-1}(f; \boldsymbol\xi) \frac{\partial \mathbf{P}(f; \boldsymbol\xi)}{\partial\xi_i}\mathbf{P}^{-1}(f; \boldsymbol\xi) \frac{\partial \mathbf{P}(f; \boldsymbol\xi)}{\partial\xi_j} \right\} \mathrm{d} f
$$
