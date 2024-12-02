### 数据模型

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
将时间维度的信息展开为行向量
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

### 波束形成技术

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
幅度图像为

![ft_basis](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\ft_basis.png)

可以看到，主瓣的宽度非常之宽，所以CDF的分辨性能不好。从某种程度上，我觉得，CDF的权重想要最小化除了当前方向其他的所有方向，所以顾此失彼，最后主瓣宽度很大。

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

单独对$R_{xx}$的第一项$A R_{ss} A^H$进行分析，由于$R_{ss}$为正定矩阵，所以其列空间与A的列空间一致，同时，$\operatorname{rank}(A) = \operatorname{rank}(AR_{ss}A^H)= K$，所以对$A R_{ss} A^H$进行特征值分解，可以得到两个空间，一是特征值不为零对应的空间，称为“信号空间”；二是，特征值为零对应的空间，称为“噪声空间”。又由于$A R_{ss} A^H$本身具有的对称矩阵的特性，所以信号空间和噪声空间是正交的。

同时，若$R_{ee} = \sigma^2 I$为白噪声协方差矩阵，那么$R_{xx}$的特征值就是$A R_{ss} A^H$特征值的平移，相对应的特征空间保持不变。MUSIC根据这个特性来进行谱估计。

因为$A$的列空间是由$K$个不同方向的导向矢量张成而来，由此我们只需要搜索不同的$\theta$对应的导向矢量，判断其是否在信号空间内，或是否正交于噪声空间。

### 空域平滑

