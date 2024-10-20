### 数据模型修改

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
\mathbf{y}(t; \theta) = \mathbf{w}^{H} \mathbf{x}(t) = \mathbf{w}^{H}(\theta) \mathbf{x}(t; \theta_d)
$$
同样，输出功率为
$$
\begin{aligned}
P(\theta) &= E\left[ | \mathbf{y}(t; \theta) |^2 \right] \\
&= \mathbf{w}(\theta)^{H} E\left[ \mathbf{x}(t; \theta_d) \mathbf{x}^{H}(t; \theta_d) \right] \mathbf{w}(\theta) \\
&= \mathbf{w}(\theta)^{H} R_{xx}(\theta_d) \mathbf{w}(\theta) \\
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

### CDF

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

### MUSIC

