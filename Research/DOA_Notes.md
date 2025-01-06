### Data Model

阵列信号处理中信号的基本数据模型为（单信号源）
$$
\mathbf{x}(t) = \mathbf{a}(\theta_t) s(t) + \mathbf{e}(t)
$$
其中，$\theta_t$表示源的方向，时间$t$可以认为固定，$\mathbf{a}(\theta_t)$表示方向角度为$\theta_t$的方向列矢量（$M$阵元）
$$
\left[\mathbf{a}(\theta_t)\right]_m = \exp\left(\mathrm{j} 2 \pi \frac{d \sin\theta_t}{\lambda} (m - 1) \right)
$$
当有多个信源时，信号满足线性叠加
$$
\begin{aligned}
\mathbf{x}(t) &= \sum\limits_{k = 1}^{K}\mathbf{a}(\theta_{k}) s_k(t) + \mathbf{e}(t) \\
&= 
\begin{bmatrix}
\mathbf{a}(\theta_{1}) & \cdots & \mathbf{a}(\theta_{K})
\end{bmatrix}
\begin{bmatrix}
s_1(t) \\
\vdots \\
s_K(t)
\end{bmatrix}
 +\mathbf{e}(t) \\
 &= A \mathbf{s}(t) + \mathbf{e}(t)
\end{aligned}
$$
将时间维度的信息按行展开
$$
\begin{aligned}
X &= 
\begin{bmatrix}
\mathbf{x}(t_1) & \cdots & \mathbf{x}(t_N) 
\end{bmatrix} \\
&= A
\begin{bmatrix}
\mathbf{s}(t_1) & \cdots & \mathbf{s}(t_N) 
\end{bmatrix}  + 
\begin{bmatrix}
\mathbf{e}(t_1) & \cdots & \mathbf{e}(t_N) 
\end{bmatrix} \\
&= AS + E
\end{aligned}
$$
在各种假设下（各项同性、线性叠加、远场平行和窄带信号），DOA估计问题等价于空间频率的估计问题。

### Beamforming

天线阵列接收到的一拍数据为一个包含$M$元素的列向量$\mathbf{x}(t)$，其中包含了关于DOA的信息，重新写为$\mathbf{x}(t; \theta_t)$（为了方便假设只有一个信源）。

波束形成技术引入一个与某个方向$\theta$对应的空域滤波系数$\mathbf{w}$，或写为$\mathbf{w}(\theta)$，由此空域滤波输出为
$$
\mathbf{y}(t; \theta) = \mathbf{w}^{H} \mathbf{x}(t) = \mathbf{w}^{H}(\theta) \mathbf{x}(t; \theta_t)
$$
同样，输出功率为
$$
\begin{aligned}
P(\theta) &= E\left[ | \mathbf{y}(t; \theta) |^2 \right] \\
&= \mathbf{w}(\theta)^{H} E\left[ \mathbf{x}(t; \theta_t) \mathbf{x}^{H}(t; \theta_t) \right] \mathbf{w}(\theta) \\
&= \mathbf{w}(\theta)^{H} R_{xx}(\theta_t) \mathbf{w}(\theta) \\
&= \mathbf{w}^{H} R_{xx} \mathbf{w} \\
\end{aligned}
$$
其中，假设$E\left[ \mathbf{x}(t) \mathbf{x}^{H}(t) \right] = R_{xx}$对于所有$t$都成立。实际上
$$
\begin{aligned}
E\left[ \mathbf{x}(t) \mathbf{x}^{H}(t) \right] &= A \mathbf{s}(t)\mathbf{s}^H(t) A^H + E\left[ \mathbf{e}(t) \mathbf{e}^{H}(t) \right]\\
&= A
\begin{bmatrix}
|s_1(t)|^2 & s_1(t)s_2^*(t) & \cdots & s_1(t)s_K^*(t) \\
s_2(t)s_1^*(t) & |s_2(t)|^2 & \cdots & s_2(t)s_K^*(t) \\
\vdots & \vdots & & \vdots \\
s_K(t)s_1^*(t) & s_K(t)s_2^*(t) & \cdots & |s_K(t)|^2
\end{bmatrix}
A^H + R_{ee}
\\

&=R_{xx}
\end{aligned}
$$
$R_{xx}$中所有关于信源DOA的信息都包含在$A$中。将上式拓展为估计式
$$
\begin{aligned}
\hat{R}_{xx} &= \frac{1}{N}\sum\limits_{n = 1}^{N} \mathbf{x}(t_n) \mathbf{x}^{H}(t_n) \\
&= \frac{1}{N} X X^H
\end{aligned}
$$
同样$\hat{R}_{xx}$包含了所有的信源DOA信息。

为了方便显示出某系数空域滤波的输出功率与实际的DOA有关，记$P(\mathbf{w}(\theta); \theta_t) = P(\theta)$，表示空域滤波器系数对应$\theta$（正在搜索角度$\theta$），外部源DOA为$\theta_t$时的输出功率，希望当两个角度一致时，输出最大功率；两个角度不一致时，输出最小功率，即
$$
\begin{cases}
\max P(\mathbf{w}(\theta); \theta_t) , & \theta_t = \theta \\
\min P(\mathbf{w}(\theta); \theta_t) , & \theta_t \neq \theta 
\end{cases}
$$
可以将其视为一个无约束多目标优化的问题，其中，$\theta$为人为控制的参数，在每个优化问题求解之前，是固定的；一个$\theta_t$代表一个优化目标，所以该问题有无数个优化目标。优化最终希望得到一个权重$\mathbf{w}$，使得当信源DOA与预先确定的$\theta$一致时，输出功率最大，剩余的其他所有$\theta_t$都能达到输出功率最小。

这个优化问题无法求解，首先，对于所有的$\theta_t$，都有一个优化目标，所以有无数个优化目标。其二，$\theta_t$如果与$\theta$不一致，可能认为能够将所有不一致的$\theta_t$都放在$R_{xx}$中，但这显然做不到，由这一点，是否可以有先验知识来假设$\theta_t$会在哪里出现呢。

### CBF

一个快拍的回波信号本质上是一个复指数函数
$$
\mathbf{x}(t) = s(t)
\begin{bmatrix}
1 \\
\exp\left(\mathrm{j} 2 \pi \frac{d \sin\theta}{\lambda}\right) \\
\vdots \\
\exp\left(\mathrm{j} 2 \pi \frac{d \sin\theta}{\lambda}(M - 1) \right)
\end{bmatrix}
+ \mathbf{e}(t)
$$
由此直接根据傅里叶基的正交特性即可完成上述多目标优化的问题
$$
\mathbf{w}(\theta) = \frac{\mathbf{a}(\theta)}{\sqrt{\mathbf{a}^H(\theta)\mathbf{a}(\theta)}}
$$
所以带入即可得到谱表达式
$$
P(\mathbf{w}(\theta); \theta_t) = \frac{\mathbf{a}^H(\theta)\hat {R}_{xx}\mathbf{a}(\theta)}{\mathbf{a}^H(\theta)\mathbf{a}(\theta)}
$$
但是由于截断特性，所以傅里叶基只是近似拥有正交特性
$$
\begin{aligned}
&\frac{1}{N} \sum\limits_{n=0}^{N-1} \exp(\mathrm{j}2\pi f_1 n) \exp(-\mathrm{j} 2\pi f_2 n) \\
&= \frac{1}{N} \sum\limits_{n=0}^{N-1} \exp\left(\mathrm{j}2\pi (f_1 - f_2) n\right)  \\
&= \frac{1}{N} \cdot \frac{1 - \exp\left(\mathrm{j}2\pi (f_1 - f_2) (N - 1)\right)}{1 - \exp(\mathrm{j}2\pi (f_1 - f_2))} \\
&= \frac{1}{N} \cdot \frac{\exp\left(\mathrm{j}2\pi (f_1 - f_2) \frac{(N - 1)}{2}\right)}{\exp\left(\mathrm{j}2\pi (f_1 - f_2) \frac{1}{2}\right)} \cdot \frac{\sin\left(2\pi (f_1 - f_2) \frac{(N - 1)}{2}\right)}{\sin\left(2\pi (f_1 - f_2) \frac{1}{2}\right)} \\
&= \frac{N - 1}{N}\exp\left(\mathrm{j}2\pi (f_1 - f_2) \left(\frac{N}{2} - 1\right)\right) \cdot \mathrm{sinc}\left( 2\pi (f_1 - f_2) \frac{(N - 1)}{2} \right)
\end{aligned}
$$
CBF本质上就是使用傅里叶变换来估计数字角频率，在单源情况下，等价于最大似然估计，性能最好。

### CAPON

仍然延续着前面所述的多目标优化问题，再次写在这里
$$
\begin{cases}
\max P(\mathbf{w}(\theta); \theta_t) , & \theta_t = \theta \\
\min P(\mathbf{w}(\theta); \theta_t) , & \theta_t \neq \theta 
\end{cases}
$$
CDF直接使用傅里叶基来近似解决这个无约束多目标优化问题，CAPON将这个多目标优化问题转换为了有约束单目标优化问题
$$
\begin{cases}
\mathbf{w}^H(\theta) \mathbf{a}(\theta) = 1 \\
\min P(\mathbf{w}(\theta); \theta_t) = \min \mathbf{w}^H(\theta) R_{xx} \mathbf{w}(\theta)
\end{cases}
$$
注意，该有约束的优化问题中的$\theta_t$是固定的，取决于实际情况下得到的数据。对比多目标优化问题，有约束单目标优化问题首先将$\theta_t$与$\theta$一致时的最大化功率变为了固定增益，虽然不能得到最大化功率，但希望至少能够在波束与目标匹配的时候能有固定增益的输出；其次，为了将所有的$\theta_t$在与$\theta$不一致时，输出功率最小化的问题简化，此优化使用了收集到的数据，仅仅只对数据中蕴藏的$\theta_t$进行最小化，所以瞬间将问题的复杂程度降下来了，代价是权重在数据没到来之前是无法计算权重，但是无论是CBF还是CAPON，最后计算的是空间谱，权重只是中间变量，空间谱一定是需要数据到来之后再来计算的，所以这个代价相当于没有。

具体求解如下

经过求解，可以得到（只要记住分子就行了，分母是归一化因子）
$$
\mathbf{w}(\theta)=\frac{\hat{R}_{xx}^{-1}\mathbf{a}(\theta)}{\mathbf{a}^H(\theta)\hat{R}_{xx}^{-1}\mathbf{a}(\theta)}
$$
由此空间谱为
$$
P(\mathbf{w}(\theta); \theta_t)=\frac1{\mathbf{a}^H(\theta)\hat{R}_{xx}^{-1}\mathbf{a}(\theta)}
$$
总结一下，CBF得到的权重是希望所有其他的角度都功率最小而得到的权重，而CAPON根据到来的数据中蕴藏的特定那几个角度进行功率最小化，所以性能会更好一点，代价就是计算量的增加。

*CAPON方法相关矩阵的奇异性还需要研究*

*另一种介于CAPON和CDF之间的方法*

*实际中，CAPON的性能受$R_{ss}$是否可逆的影响很大，这是为什么呢？*
初步的想法是$R_{ss}$的不可逆使得蕴藏在$R_{xx}$中的角度信息有了损失，从而在CAPON优化的过程中不能完全抑制正真$\theta_d$的信号，所以使得CAPON的性能变差了。

### 参数估计视角下的CAPON（MVDR）

如果目标信号的角度已知为$\theta_s$，且空间噪声为高斯白噪声，模型如下
$$
\mathbf{x}(t) = \mathbf{a}(\theta_s) s(t) + \mathbf{n}(t)
$$
该数据模型为线性模型，再加上噪声为高斯白噪声，所以其MVU估计量为线性的形式
$$
\hat{s(t)} = \mathbf{w}^{H} \mathbf{x}(t) = \frac{\mathbf{a}^H(\theta_s) }{\mathbf{a}^H(\theta_s) \mathbf{a}(\theta_s)} \mathbf{x}(t) = \frac{\mathbf{e}^H }{\mathbf{e}^H\mathbf{e}} \mathbf{x}(t)
$$
现在假设来了一个已知信号的干扰信号，方向为$\theta_j$，则模型如下
$$
\begin{aligned}
\mathbf{x}(t) &= \mathbf{a}(\theta_s) s(t) + \mathbf{a}(\theta_j) s_j(t) +  \mathbf{n}(t) \\
&= \mathbf{a}(\theta_s) s(t) + \mathbf{n}_1(t)
\end{aligned}
$$
其中，$s_j(t)$是与噪声独立的WSS随机过程，其方差满足各态历经性，所以可以通过时间平均的方差来估计某一时刻的方差。我们将干扰信号归于噪声中，故此时的模型变为色噪声下的线性模型，色噪声的协方差矩阵为
$$
C = E\left[ \mathbf{n}_1(t) \mathbf{n}_1^H(t) \right] = P \mathbf{i} \mathbf{i}^H + \sigma^2 I
$$
在色噪声下的线性模型同样拥有闭式的线性MVU表达式
$$
\hat{s(t)} = \frac{\mathbf{e}^H C^{-1}}{\mathbf{e}^H C^{-1} \mathbf{e}} \mathbf{x}(t)
$$
线性权重则为
$$
\mathbf{w}_{\text{opt}} = \frac{C^{-1} \mathbf{e}}{\mathbf{e}^H C^{-1} \mathbf{e}} = c_1 \mathbf{e} - c_2 \mathbf{i}
$$
其中
$$
C^{-1} = \frac{1}{\sigma^2}\left( I -  \frac{P}{MP + \sigma^2} \mathbf{i} \mathbf{i}^H  \right) \\
c_1 = \frac{1}{M - \frac{P|\mathbf{e}^H \mathbf{i}|^2}{MP + \sigma^2}} \\
c_2 = \frac{P \mathbf{i}^H \mathbf{e} }{M(MP + \sigma^2) - P |\mathbf{e}^H \mathbf{i}|^2}
$$
将这个权重带入目标信号中，可以得到
$$
\mathbf{w}_{\text{opt}}^H\cdot s(t)\mathbf{e} = s(t)
$$
目标信号无失真的恢复了，这是源于MVU的无偏限制。

对于干扰信号，可以得到
$$
\mathbf{w}_{\text{opt}}^H\cdot s_j(t)\mathbf{i} = (c_1 \mathbf{e}^H \mathbf{i} - M c_2^*) s_j(t)
$$
干扰信号在功率上损失了$|c_1 \mathbf{e}^H \mathbf{i} - M c_2^*|^2$倍。

如果干扰信号与目标信号方向重合，即$\mathbf{e} = \mathbf{i}$，则$|c_1 \mathbf{e}^H \mathbf{i} - M c_2^*|^2 = 1$，代表着对干扰信号没有能量的衰减。

![image-20241028154353906](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20241028154353906.png)

如图表示当目标信号位于$90 \degree$时，干扰信号的方向与能量衰减之间的关系。可以看到，当干扰信号与目标信号的干扰重合时，是不会对干扰信号造成任何衰减的。

所以对于CAPON来说，在每一个正在搜索的角度$\theta$，将此时所有的其他$K$个方向的信号都视为干扰，放入噪声项中，最后得到能够抑制所有$K$个信号的权重，若当前搜索的方向没有与$K$个方向重合，那么这$K$个方向的信号都得到了最大的抑制。当搜索的方向与这$K$个方向的信号有一个重合时，那么重合的信号就不会被抑制，从而达到了搜索的目的。

注意，CAPON中包含的MVU估计量是对信号的估计，并不是对角度的估计，所以这里的MVU估计量并不代表着方向的估计是最佳的。

同时，也应该意识到，即使是对信号估计的性能，在多个信源的情况下，当搜索角度与其中一个信源重合时，此时得到对信号的估计并不是MVU估计量，因为当对该角度的信号进行估计时，考虑的噪声协方差矩阵包含了该角度的信号，这是多余的。而对于CBF来说，其对信号的估计是在假设噪声为高斯白噪声时的MVU估计量，所以当信源只有一个时，其对信号估计的性能就是最优的。

### MUSIC

计算接收信号的协方差矩阵
$$
\begin{aligned}
R_{xx} &= E\left[ \mathbf{x}(t) \mathbf{x}^H(t) \right] \\
&= E\left[ \left(A\mathbf{s}(t) + \mathbf{e}(t)\right) \left(A\mathbf{s}(t) + \mathbf{e}(t)\right)^H \right] \\
&= A E\left[ \mathbf{s}(t) \mathbf{s}^H(t)\right] A^H + E\left[ \mathbf{e}(t) \mathbf{e}^H(t)\right] \\
&= A R_{ss} A^H + R_{ee}
\end{aligned}
$$
这里介绍一个结论，即$AA^H$与$A$有相同的列空间。

单独对$R_{xx}$的第一项$A R_{ss} A^H$进行分析，由于$R_{ss}$为正定矩阵，所以其列空间与$A$的列空间一致，同时，$\operatorname{rank}(A) = \operatorname{rank}(AR_{ss}A^H)= K$，所以对$A R_{ss} A^H$进行特征值分解，可以得到两个空间，一是特征值不为零对应的空间，称为“信号空间”；二是特征值为零对应的空间，称为“噪声空间”。又由于$A R_{ss} A^H$为对称矩阵，所以信号空间和噪声空间是正交的。

同时，若$R_{ee} = \sigma^2 I$为白噪声的协方差矩阵，那么$R_{xx}$的特征值就是$A R_{ss} A^H$特征值的平移，对应的特征空间保持不变。MUSIC根据这个特性来进行谱估计。

因为$A$的列空间是由$K$个不同方向的导向矢量张成而来，由此我们只需要搜索不同的$\theta$对应的导向矢量，判断其是否在信号空间内，或是否正交于噪声空间。

spectral-music和root-music。

由上面的推导可知，MUSIC所需要两个必要的条件是

- 白噪声假设，由此才能使得信号空间在加上噪声之后仍然保持不变；
- 信源无关假设，使得$A$的列空间与$A R_{ss} A^H$的列空间相同。

*有一篇论文记得看，Stoica写的。*

### ESPRIT



### Forward/Backward Smoothing

$R_{ss}$非满秩造成的影响。

### Maximum Likelihood

将接收数据$X$进行拉伸运算，等价于$\left\{ \mathbf{x}(t) \right\}_{t = 1}^{N}$排列为列向量的形式
$$
\begin{aligned}
\mathbf{x} &= \operatorname{vec}(X) \\
&= \operatorname{vec}\left(AS + E\right) \\
&= \left( I \otimes A  \right) \operatorname{vec}(S) + \operatorname{vec}(E) \\
&= G \mathbf{s} + \mathbf{e} \\
\end{aligned}
$$
所以拉长后观测数据的对数似然函数为
$$
\ln p(\mathbf{x}; \boldsymbol{\xi}) = c - \sigma^2 (\mathbf{x} - G \mathbf{s})^H (\mathbf{x} - G \mathbf{s})
$$
其中假设噪声为方差为$\sigma^2$的白噪声。待估计的参数为（确定性假设的波形拆分为了实部和虚部）
$$
\boldsymbol{\xi} = \begin{bmatrix}
\theta_1 & \cdots & \theta_K & \bar{\mathbf{s}}^T & \tilde{\mathbf{s}}^T
\end{bmatrix}^T
$$
观察数据模型啊，可知该模型包含了线性部分的参数$\bar{\mathbf{s}}$和$\tilde{\mathbf{s}}$，以及非线性部分的参数$\boldsymbol{\theta}$，所以可以采用参数分离的技巧，即
$$
\hat{\mathbf{s}} = \left(G^H G \right)^{-1}G^H \mathbf{x}
$$
由此最大化似然等价于最小化一下目标函数
$$
\begin{aligned}
J &= (\mathbf{x} - G \hat{\mathbf{s}})^H (\mathbf{x} - G \hat{\mathbf{s}}) \\
&= \left[\mathbf{x} - G \left(G^H G \right)^{-1}G^H \mathbf{x}\right]^H \left[\mathbf{x} - G \left(G^H G \right)^{-1}G^H \mathbf{x}\right] \\
&= \left(P_G^{\perp} \mathbf{x}\right)^H\left(P_G^{\perp} \mathbf{x}\right) \\
&= \mathbf{x}^H P_G^{\perp} \mathbf{x} \\
&= \sum\limits_{t = 1}^N \mathbf{x}^H(t) P_A^{\perp} \mathbf{x}(t)
\end{aligned}
$$
等价于最大化$\sum\limits_{t = 1}^N \mathbf{x}^H(t) P_A \mathbf{x}(t)$或者$\operatorname{tr} \left( P_A^{\perp} \hat{R}_{xx}  \right)$，注意前者的表达式可以理解为$\mathbf{x}(t)$向量投影到$A$列空间后向量的模长平方。

由于非线性的影响，目标函数具有复杂的多峰形状（非凸）。在实际的操作中，我们以各种各样的方式多维搜索$\boldsymbol\theta$（比如网格搜索），来使得对应的目标函数进行最优化。

### Expectation-Maximization

参考：Theory and Use of the EM Algorithm. 

Parameter estimation of superimposed signals using the EM algorithm. 

Maximum-likelihood narrow-band direction finding and the EM algorithm. 

对于DOA估计的最大似然求解问题，Maximum-Likelihood DOA Estimation by Data-Supported Grid Search这篇文章将求解方法按照计算复杂度从低到高分为四类：大样本高信噪比近似方法、局部搜索方法、全局搜索方法以及网格搜索法。由于近似方法得到的估计量本质上不属于最大似然估计的范畴了，所以本节不考虑。局部搜索方法即使用优化理论的方法以似然函数作为优化的目标函数，常见的有EM、梯度类、ADMM等方法，本节考虑EM算法。

EM算法的步骤和基本原理参考Theory and Use of the EM Algorithm，简单来说，EM算法使用MM准则，通过对似然函数在某次迭代的参数值的下界进行优化，使得似然函数在迭代的参数值下具有单调不降的特性，由此收敛到似然函数的稳定点。

可以将EM算法具体运用在多信号叠加参数估计的问题上，研究的信号模型为
$$
\mathbf{y}(t) = \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) + \mathbf{n}(t)
$$
其中，$\mathbf{y}(t)$可以为任意维的列向量（当然也包括标量），$\boldsymbol{\theta}$为该模型下待估计的所有参数，$\mathbf{s}_k(t; \boldsymbol{\theta})$表示从全局参数$\boldsymbol{\theta}$映射到第$k$个时域信号的映射，$\mathbf{n}(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{Q})$。直观来说，观测的数据由多个信号以及噪声叠加而成，其中每个信号都可以由一个全局的参数$\boldsymbol{\theta}$以各自不同的映射$\mathbf{s}_k(t; \boldsymbol{\theta})$来产生。如果这个模型满足控制每个信号源的参数独立，不会相互影响，也即将参数$\boldsymbol{\theta}$分开为$K$个互不重叠的参数$\boldsymbol{\theta}_k$，第$k$个信号源由$\boldsymbol{\theta}_k$独立产生，那么就变为了论文中的模型
$$
\mathbf{y}(t) = \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}_k) + \mathbf{n}(t)
$$
暂时先考虑第一种具有全局参数的模型，似然函数为
$$
\ln p(\mathbf{y}; \boldsymbol{\theta}) = c_1 - \sum\limits_{t = 1}^{N} \left[ \mathbf{y}(t) -  \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\mathbf{y}(t) -  \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) \right]
$$
根据EM算法，我们要指定一个隐变量（complete data）$\mathbf{x}$，其与参数$\boldsymbol{\theta}$，观测数据$\mathbf{y}$应该满足$\boldsymbol{\theta} \rightarrow \mathbf{x} \rightarrow \mathbf{y}$的Markov链的关系，也即$\mathbf{y}$只取决于$\mathbf{x}$而不取决于参数$\boldsymbol{\theta}$。在多信号叠加的模型下，我们指定隐变量为每个单独的信号源外加一个噪声
$$
\mathbf{x}(t) = \begin{bmatrix}
\mathbf{x}_1(t) \\
\vdots \\
\mathbf{x}_K(t)
\end{bmatrix} \\
\mathbf{x}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}) + \mathbf{n}_k(t)
$$

$$
\mathbf{y}(t) = \begin{bmatrix}
\mathbf{I} & \cdots & \mathbf{I}
\end{bmatrix}
\mathbf{x}(t)  = \mathbf{H} \mathbf{x}(t)
$$

其中，假设每个子噪声$\mathbf{n}_k(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{Q}_k)$为相互独立，则满足$\mathbf{Q}_k = \beta_k \mathbf{Q}$，且$\sum\limits_{k = 1}^{K} \beta_k = 1$。则可以写出隐变量的似然函数
$$
\begin{aligned}
\ln p(\mathbf{x}; \boldsymbol{\theta}) &= c_2 - \sum\limits_{t = 1}^{N} \left[ \mathbf{x}(t) -  \mathbf{s}(t; \boldsymbol{\theta}) \right]^H \mathbf{\Lambda}^{-1} \left[\mathbf{x}(t) -  \mathbf{s}(t; \boldsymbol{\theta}) \right] \\
&= c_2 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \mathbf{x}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\mathbf{x}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right] \\
&= c_3 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[
-\mathbf{x}^H_k(t) \mathbf{Q}^{-1}\mathbf{s}_k(t; \boldsymbol{\theta}) - \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{x}_k(t) + \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{s}_k(t; \boldsymbol{\theta})
\right]
\end{aligned}
$$

$$
\mathbf{\Lambda} = \begin{bmatrix}
\mathbf{Q}_1 & & \\
& & \ddots & \\
& & & \mathbf{Q}_K
\end{bmatrix}
$$

在EM算法进行E-step时，需要计算隐变量似然函数的条件期望（假设参数迭代到了$\boldsymbol{\theta}^{(n)}$）
$$
\begin{aligned}
& E_{\mathbf{x} \mid \mathbf{y}} \left[\ln p(\mathbf{x}; \boldsymbol{\theta})  ; \boldsymbol{\theta}^{(n)} \right] \\
&= c_4 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[
-\hat{\mathbf{x}}^H_k(t) \mathbf{Q}^{-1}\mathbf{s}_k(t; \boldsymbol{\theta}) - \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \hat{\mathbf{x}}_k(t) + \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{s}_k(t; \boldsymbol{\theta}) \right] \\
&= c_5 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]
\end{aligned}
$$
其中
$$
\begin{aligned}
\hat{\mathbf{x}}(t) &= E_{\mathbf{x} \mid \mathbf{y}} \left[\mathbf{x}(t) ; \boldsymbol{\theta}^{(n)} \right] \\
&= \mathbf{x}(t) + \mathbf{\Lambda} \mathbf{H}^T \left[ \mathbf{H} \mathbf{\Lambda} \mathbf{H} \right]^{-1} \left[ \mathbf{y} - \mathbf{H} \mathbf{s}(t; \boldsymbol{\theta}) \right]
\end{aligned}
$$
展开后可得到
$$
\hat{\mathbf{x}}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}_k) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K} \mathbf{s}_l(t; \boldsymbol{\theta}_l) \right]
$$
由此即可完成E-step的计算。

在M-step中，我们需要优化的目标函数为$E_{\mathbf{x} \mid \mathbf{y}} \left[\ln p(\mathbf{x}; \boldsymbol{\theta})  ; \boldsymbol{\theta}^{(n)} \right]$，由E-step可知，其相当于在参数为$\boldsymbol{\theta}^{(n)}$时计算出了隐变量（complete data）的观测数据$\hat{\mathbf{x}}$，然后再在求出隐变量模型的最大似然解，隐变量模型为
$$
\mathbf{x}(t) = \begin{bmatrix}
\mathbf{s}_1(t; \boldsymbol{\theta})  \\
\vdots \\
\mathbf{s}_K(t; \boldsymbol{\theta}) 
\end{bmatrix} +  \begin{bmatrix}
\mathbf{n}_1(t)  \\
\vdots \\
\mathbf{n}_K(t)

\end{bmatrix}
$$
综上所述，EM算法对于多源信号叠加问题的求解步骤为（假设参数迭代到了$\boldsymbol{\theta}^{(n)}$）

- E-step：对于所有$k$，计算隐变量的观测数据$\hat{\mathbf{x}}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}_k^{(n)}) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K} \mathbf{s}_l(t; \boldsymbol{\theta}_l^{(n)}) \right]$；
- M-step：在隐变量的模型上求解其的最大似然估计$\boldsymbol{\theta}^{(n + 1)} = \arg \max\limits_{\boldsymbol{\theta}} \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]$。

EM算法是否能够简化运算的关键在于M-step的最大似然估计问题是否相较于观测变量是否更加简单，从直觉来讲，多源信号叠加模型的隐变量的设置将叠加在一起的时域信号拆开，由此隐变量的的估计问题是多源叠加信号拆开为单源信号后进行的，一般而言，单源信号的估计问题要比多源信号估计问题要更加简单。同时，M-step的运算并不要求一定要是最大似然估计，只要比原来更好就行，这样也能保证EM算法具有单调不减的特性，也就是Generalized EM算法。

如果假设每个源的信号由单独且互不重叠的参数控制，那么M-step的多维参数估计问题可以拆开为并行的每个源单独的参数估计问题，M-step修改为

- M-step：对于所有$k$，在隐变量的模型的单源上求解其的最大似然估计$\boldsymbol{\theta}^{(n + 1)}_k = \arg \max\limits_{\boldsymbol{\theta}_k} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}_k) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}_k) \right]$。

相较于全局共享的参数，每个源的参数估计问题可以并行单独处理。

**EM algorithm for DOA estimation**

现在考虑将EM算法运用在DOA估计问题上，其满足每个源都由独立的参数控制的条件
$$
\mathbf{s}_k(t; \boldsymbol{\xi}_k) = \mathbf{a}(\theta_k) s_k(t) \\
\boldsymbol{\xi}_k = \begin{bmatrix}
\theta_k & \bar{\mathbf{s}}_k^T & \tilde{\mathbf{s}}_k^T
\end{bmatrix}
$$
所以算法步骤为

- E-step：对于所有$k$，计算隐变量的观测数据$\hat{\mathbf{x}}_k(t) =  \mathbf{a}(\theta_k^{(n)}) s_k^{(n)}(t) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K}  \mathbf{a}(\theta_l^{(n)}) s_l^{(n)}(t) \right]$；
- M-step：对于所有$k$，在隐变量的模型的单源上求解其的最大似然估计$\boldsymbol{\xi}^{(n + 1)}_k = \arg \max\limits_{\boldsymbol{\xi}_k} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{a}(\theta_k) s_k(t) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{a}(\theta_k) s_k(t) \right]$。

其中，可以在EM中使用分离线性参数的技巧（线性参数可有非线性参数表示），使得收敛速度更快。即在M-step中，我们不需要正真找到线性参数的最大似然估计量，只需要找到非线性参数的最大似然估计，而对于单源DOA估计的最大似然估计，就是周期功率谱的最大频率点。在求出所有非线性参数后，本轮得到的线性参数的最大似然估计就是非线性参数的线性变换，由此可以重新改写M-step

- M-step：对于所有$k$，在隐变量的模型的单源上求解非线性参数的最大似然估计$\theta^{(n + 1)}_k = \arg \max\limits_{\theta_k} \sum\limits_{t = 1}^{N} \| \mathbf{a}^H(\theta_k) \mathbf{x}(t) \|^2$。在得到了所有源的DOA估计后，估计线性参数$\mathbf{s}^{(n)}(t) = (\mathbf{A}^H\mathbf{A})^{-1} \mathbf{A}^H \mathbf{y}(t)$。

