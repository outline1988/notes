### Wideband DOA

先不考虑任何近似，一个$M$阵元的线阵接收到来自空间的$K$个信号，则阵列的第$m$阵元接收到的信号为
$$
y_m(t) = \sum\limits_{k = 1}^{K}  s_k(t - \tau_{m, k}) + n_m(t)
$$
其中，$\tau_{m, k}$表示第$m$个阵元所接收到第$k$个信号所要经历的延迟。以第一个阵元接收到的信号作为参考，则$\tau_{1, k} = 0$，对于所有$k$成立。

对该式的两边同时傅里叶变换，转换为频域
$$
\begin{aligned}
Y_m(f) &= \sum\limits_{k = 1}^{K}  S_k(f) \exp(-\mathrm{j} 2 \pi f \tau_{m, k}) + N_m(f) \\
&= \begin{bmatrix}
\exp(-\mathrm{j} 2 \pi f \tau_{m, 1}) & \cdots & \exp(-\mathrm{j} 2 \pi f \tau_{m, K})
\end{bmatrix}
\begin{bmatrix}
S_1(f) \\ \vdots \\ S_K(f)
\end{bmatrix} + N_m(f)
\end{aligned}
$$
所以，将所有阵元的接收信号排列为向量
$$
\begin{aligned}
\mathbf{Y}(f) &= \begin{bmatrix}
\exp(-\mathrm{j} 2 \pi f \tau_{1, 1}) & \cdots & \exp(-\mathrm{j} 2 \pi f \tau_{1, K}) \\
\vdots & \ddots & \vdots \\
\exp(-\mathrm{j} 2 \pi f \tau_{M, 1}) & \cdots & \exp(-\mathrm{j} 2 \pi f \tau_{M, K})
\end{bmatrix} \begin{bmatrix}
S_1(f) \\ \vdots \\ S_K(f)
\end{bmatrix} + \mathbf{N}(f) \\
&= \sum\limits_{k = 1}^K \mathbf{a}(f, \theta_k) S_k(f) + \mathbf{N}(f) \\
&= \mathbf{A}(f, \boldsymbol{\theta}) \mathbf{S}(f) + \mathbf{N}(f)
\end{aligned}
$$
其中，对于均匀线阵，$\tau_{m, k} = (m - 1)d \sin( \theta_k)  / c$。

上式即为不含任何近似的DOA估计表达式，若为窄带信号，则可以进一步进行近似，由此进行化简。假设$\mathbf{S}(f)$的带宽范围为$[f_L, f_H]$，即$\mathbf{S}(f)$只在这个范围内有值，其他地方为0。若这个范围很小，我们可以认为$\mathbf{A}(f, \boldsymbol{\theta})$的变化很小，即
$$
\mathbf{A}(f_L, \boldsymbol{\theta}) \approx \mathbf{A}(f_H, \boldsymbol{\theta})
$$
由此近似为$\mathbf{A}(f, \boldsymbol{\theta}) \approx \mathbf{A}(f_0, \boldsymbol{\theta})$。从量化（频域）的角度来讲，$2 \pi f D / c$需要在$[f_L, f_H]$的范围内变化很小，即
$$
2 \pi (f_H - f_L) D / c \leq 1  \rightarrow B D / c \ll 1
$$
从时域的角度来说，孔径渡越时间最大可能为$D / c$，孔径渡越时间不会使得带宽$B$的信号有很大的影响，所以就有$B D / c \ll 1$。这就是窄带信号的条件，最终得到窄带的频域和时域表示为
$$
\mathbf{Y}(f) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{S}(f) + \mathbf{N}(f) \\
\mathbf{y}(t) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{n}(t)
$$
也就是说，窄带DOA估计实际上只是宽带DOA的一种特殊的情况。

#### Incoherent Signal Subspace Method

对于宽带DOA估计，最自然的想法就是将一个宽带的信号经过窄带滤波器组，形成若干个窄带信号
$$
\mathbf{Y}_i = \mathbf{A}(f_i, \boldsymbol{\theta}) \mathbf{S}_i + \mathbf{N}_i
$$
对每一个窄带模型进行对应的谱估计，则最终宽带信号的谱就是所有窄带信号谱之和。频率估计就在宽带信号谱上进行搜索。

对宽带信号进行滤波通常是是基于DFT直接置零进行的，DFT直接置零的效果可以看其他笔记。假设通过滤波器组得到了$P$个长度与原来相等都为$N$的信号，则使用MUSIC方法宽带谱为
$$
P(\theta) =  \frac{1}{\sum\limits_{i = 0}^{P - 1} \mathbf{a}^H(f_i, \theta) \mathbf{W}_i \mathbf{W}_i^H \mathbf{a}(f_i, \theta)} 
$$

### 窄带模型的空间变换

对于某一窄带DOA模型
$$
\mathbf{y}(t) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{n}(t)
$$
在白噪声的假设下，子空间类方法要求利用协方差矩阵
$$
\mathbf{R} = E\left[ \mathbf{y}(t) \mathbf{y}^H(t) \right] = \mathbf{A}(\boldsymbol{\theta}) \mathbf{R}_{ss} \mathbf{A}^H(\boldsymbol{\theta}) + \sigma^2 \mathbf{I}
$$
在信源协方差矩阵满秩的情况下，利用协方差矩阵的信号空间与阵列流形$\mathbf{A}(\boldsymbol{\theta})$的列空间相同的特性，来筛选出处于信号空间的DOA。

实际上，子空间方法只利用了$\mathbf{A}(\boldsymbol{\theta})$的列空间特性，若对$\mathbf{y}(t)$进行某一已知可逆转换
$$
\mathbf{y}'(t) = \mathbf{T}\mathbf{y}(t) = \mathbf{T} \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{T}\mathbf{n}(t)
$$
其中，$\mathbf{T}$是可逆方阵，其将原来$\mathbf{A}$的列空间转换为了另一子空间，该子空间的各个基如下，且相互线性独立（$\mathbf{T} \mathbf{A}$为满秩矩阵）
$$
\mathbf{T} \mathbf{A}(\boldsymbol{\theta}) =  \begin{bmatrix}
\mathbf{T}\mathbf{a}(\theta_1) & \cdots & \mathbf{T}\mathbf{a}(\theta_K)
\end{bmatrix}
$$
由此形成的$\mathbf{y}'(t)$的协方差矩阵
$$
\mathbf{R}' = E\left[ \mathbf{y}'(t) \mathbf{y}'^H(t) \right] = \mathbf{T} \mathbf{A}(\boldsymbol{\theta}) \mathbf{R}_{ss} \mathbf{A}^H(\boldsymbol{\theta}) \mathbf{T}^H + \sigma^2 \mathbf{T} \mathbf{T}^H
$$
若$\mathbf{T}$为正交矩阵，转换后的噪声仍然是白噪声$\sigma^2 \mathbf{T} \mathbf{T}^H = \sigma^2 \mathbf{I}$，则仍然可以用子空间方法来进行DOA估计。此时通过协方差$\mathbf{R}'$得到的信号空间为$\mathbf{T}\mathbf{A}(\boldsymbol{\theta})$的列空间，此时产生搜索角度的导向矢量时，还要再进行一个已知的变换$\mathbf{T}\mathbf{a}(\theta)$，来观察其是否处于阵列流形的列空间中。

在经过变换后，得到的协方差矩阵$\mathbf{R}'$的信号空间为$\mathbf{T} \mathbf{A}(\boldsymbol{\theta})$的列空间，而相应的噪声空间就是其的正交。所以在搜索时，首先会产生一个导向矢量，然后再经过变换得到另一个矢量$\mathbf{T}\mathbf{a}(\theta)$，接着判断该向量是否位于信号空间中（或是正交于噪声空间）。首先如果搜索的角度$\theta$刚好匹配上了真值$\boldsymbol{\theta}$，产生的搜索向量一定位于信号空间（或者正交于噪声空间）。如果搜索的角度$\theta$与真值不匹配，那么转换后的向量希望其能尽量尽可能位于噪声空间。

如果不进行转换（或者转换矩阵就是一个单位阵），也就是原本子空间类方法的例子。那么由ULA导向矢量本质是复指数函数的性质，则不匹配的搜索角度$\theta$基本上是位于噪声空间的（原则上阵元个数是无穷时，不匹配的角度是严格位于噪声空间的），子空间类方法超分辨的本领最直接的相关的就是真值的各个导向矢量之间的相关性如何。这边的讨论是关于原本不进行转换后的子空间类方法的讨论，并可以结合最大似然函数中投影的性质来讨论，这个问题留在之后再来想，而且还有一篇stoica的论文需要看。

若经过转换，首先匹配真值$\boldsymbol{\theta}$的搜索角度一定会位于信号空间（即正交于噪声空间），问题在于，不匹配的角度是否能够尽量位于噪声空间（即正交信号空间），这取决于转换$\mathbf{T}$的性质。先来讨论转换矩阵$\mathbf{T}$为正交矩阵时，原本两个角度的相关$\mathbf{a}^H(\theta_1) \mathbf{a}(\theta_2)$在经过变换后的两个矢量的相关为
$$
\left[ \mathbf{T} \mathbf{a}(\theta_1) \right] ^H \left[ \mathbf{T} \mathbf{a}(\theta_2) \right] = \mathbf{a}^H(\theta_1) \mathbf{T}^H \mathbf{T} \mathbf{a}(\theta_2) = \mathbf{a}^H(\theta_1)\mathbf{a}(\theta_2)
$$
可以发现，不同角度之间的正交性没有影响。所以使用正交矩阵的变换后，并不会对子空间类方法带来任何影响。

*是否可以经过一个已知变换，在特定区域角度的距离拉大，从而在该区域实现超超分辨。*

#### 多窄带模型相参估计

若对于同一波达方向$\boldsymbol{\theta}$，有不同的窄带模型，同一写为如下
$$
\mathbf{X}_i = \mathbf{A}_i(\boldsymbol{\theta}) \mathbf{S}_i + \mathbf{N}_i, \quad i = 1, \cdots , L
$$
对于每一个$i$，相同点只有阵列流行$\mathbf{A}_i$是由相同$\boldsymbol{\theta}$产生的，以及$\mathbf{N}_i$是白噪声。而不同$i$的阵列流行是不同，快拍数也有不同。目标是进行coherent处理以估计DOA，基本的思路是对每一个$i$都进行某一个已知变换$\mathbf{T}_i$，则将子空间转换到了$\mathbf{T}_i \mathbf{A}$，讲转换后的数据协方差加在一起
$$
\mathbf{R} = \sum\limits_{i = 1}^L \mathbf{R}_i'
$$
协方差加在一起的体现了coherent处理，利用combine的协方差来进行子空间类的DOA操作。

最显然的思路就是，对于每一个$i$的转换（$\mathbf{T}_i$为正交矩阵），若$\mathbf{T}_i \mathbf{A}_i = \mathbf{Q}$转换到了同一矩阵，这个信号子空间$\mathbf{Q} = \mathbf{Q}(\boldsymbol{\theta})$是由$\theta_1 , \cdots ,\theta_K$经过某一已知特定变换形成$K$个列向量，而张成出的子空间。如此，则可以利用$\mathbf{R}$来进行估计，此时
$$
\begin{aligned}
\mathbf{R} &= \sum\limits_{i = 1}^L \mathbf{R}_i' \\
&= \sum\limits_{i = 1}^L \mathbf{T}_i \mathbf{A}_i(\boldsymbol{\theta}) \mathbf{R}_{ss, i} \mathbf{A}_i^H(\boldsymbol{\theta}) \mathbf{T}_i^H + L \sigma^2 \mathbf{I} \\
&= \mathbf{Q} \mathbf{R}_{ss}' \mathbf{Q}^H + L \sigma^2 \mathbf{I}
\end{aligned}
$$
问题的关键在于如何找到$\mathbf{T}_i$。经过$\mathbf{T}_i$后，只能转换成同一矩阵$\mathbf{Q}$，同一列空间但是不同矩阵是不行的，因为经过加和后，列空间可能相互抵消，称为列空间的子集。

这里还有些问题值得探讨，其一，真值$\theta_1 , \cdots ,\theta_K$经过变换所形成的$K$个向量是否能够落入到$\mathbf{Q}$空间中，这一点应该是毋庸置疑的；其二，除此之外的角度是否能够尽量落到$\mathbf{Q}$的正交空间中，也就是信号子空间对于真值DOA敏感程度的问题；其三，是否能够定制化$\mathbf{Q}$空间，使得其对特定范围的角度敏感，由此在该特定范围有利于实现超分辨。

是否可以将一个窄带模型，但是信源相关，滤波成两个窄带模型，但是两个窄带模型加在一起会使得信源去相关，可能需要超窄的两个滤波器。

### Coherent Signal Subspace Method

对于宽带问题，在使用频率分段的方法讲宽带数据分成多个窄带数据时，可以应用前面提到的多窄带模型相参DOA估计，对于一个宽带问题，得到的多个窄带模型
$$
\mathbf{Y}_i = \mathbf{A}(f_i, \boldsymbol{\theta}) \mathbf{S}_i + \mathbf{N}_i
$$


对于这种频率分段而形成的多个窄带模型，对其进行转换的矩阵称为聚焦矩阵，聚焦矩阵的其中一种实现方式称为信号子空间转换（signal subspace transformation, SST）矩阵。SST矩阵的具体实现很多种，但都满足如下表达式
$$
\mathbf{T}_i = \mathbf{Q}_r \mathbf{Q}_i^H
$$
其中，$\mathbf{Q}_i \in \mathbb{C}^{M \times M}$是正交矩阵，前$K$列张成的子空间与$\mathbf{A}(f_i, \boldsymbol{\theta})$的列空间相同。同理，$\mathbf{Q}_r$的前$K$列张成的子空间与$\mathbf{A}(f_r, \boldsymbol{\theta})$相同。具体的证明参考文章：On focusing matrices for wide-band array processing。

注意到$\mathbf{T}_i$是正交矩阵，所以其对于单个窄带模型来说，不会对子空间类方法的性能造成什么影响。但是其将所有窄带数据不同的信号空间（阵列流形张成的空间）都转换为了同一信号空间，完成的方式是先用$\mathbf{Q}_i^H$对原先的阵列流形$\mathbf{A}_i$转换至单位阵空间，再用$\mathbf{Q}_r$将单位阵空间转换为$\mathbf{A}_r$空间。由于需要提前得到关于$\mathbf{A}_r$和$\mathbf{A}_i$的子空间信息以形成相应的$\mathbf{Q}_r$和$\mathbf{Q}_i$，所以需要知道信源的真值，在实践中，这个真值由预先的测量值代替。两个正交矩阵都可以由对于阵列流形的QR分解来得到。同时RSS矩阵也是SST矩阵的一种特殊情况。

这种讲阵列流形的信号空间转换的思路，不仅仅局限于频率分段的窄带模型，而可以应用到更广的不同阵列流形窄带模型的相参合成中。

在实际的应用中，不同的聚焦矩阵会影响最后的性能，有什么影响尚待研究。同时由于CSSM的基本思想在于多个协方差矩阵的平均，所以会产生类似于空域平滑的好处：信源去相关；快拍数过少导致协方差矩阵的秩不足信源个数。

### BI-CSSM

论文：Efficient wideband source localization using beamforming invariance technique

除了基于子空间转换的CSSM方法，还有另一种将复指数向量为基所形成的空间转换为单位向量为基的波束空间的转换方法。首先从单个窄带模型的波束空间转换说起。

#### 窄带模型的波束空间转换

对于某一个窄带模型
$$
\mathbf{y}(t) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{n}(t)
$$
考虑这样一个$M \times P$的矩阵
$$
\mathbf{W} = \begin{bmatrix}
\mathbf{w}_1 & \cdots & \mathbf{w}_P

\end{bmatrix}
$$
其中，$\mathbf{w}_p$是一个指向$p$方向的波束形成器，共有$P$个不同方向的波束形成器。将适用该矩阵对窄带数据进行转换
$$
\begin{aligned}
\mathbf{W}^H \mathbf{y}(t)& = \mathbf{W}^H \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{n}(t) \\
&= \mathbf{B}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{n}_B(t)
\end{aligned}
$$
所形成了一个新的波束空间的阵列流形$\mathbf{B}(\boldsymbol{\theta}) \in \mathbb{C}^{P \times K}$。阵列流形$\mathbf{A}$的空间很好理解，就是不同角度所形成的不同频率的复指数函数张成的空间。对于波束空间阵列流形$\mathbf{B}(\boldsymbol{\theta})$来说，每行代表不同的波束指向，每列代表对于某一个角度来说，不同波束指向所形成的相应。理想来说，我们希望波束形成器在指定方向增益最高，其他方向增益最低。所以，在$\mathbf{B}(\boldsymbol{\theta})$中对于某一个角度的列来说，其只在匹配上的波束形成器形成响应，而在其他方向的波束形成器上为0。换句话说，其将复指数向量为基的空间转换为单位向量为基的空间，由此，在后续的子空间类方法中，都在波束空间中进行（只不过波束空间是经过原先的复指数空间而来）。

从直观的角度来说，对于某一个方向的的来波，我们相当于在不同方向设置了多个天线，每个天线都会对该方向做出相应，方向匹配的天线响应大，不匹配的响应小，甚至没有匹配。波束空间的方法就是通过多个空间的响应，来找到区分出某一来波的方向。

波束空间的方法还有许多地方需要与复指数空间的方法进行对比和讨论。比如，波束天线的指向和个数该如何选择，又会怎样影响性能。目前来看，波束天线的指向可以集中在某一个范围中，以进一步的对波束进行锐化，这样是否能够提高分辨能力呢（瞎想的）。波束天线数量的选择，要大于信源个数（子空间方法的要求），同时要小于等于阵元个数（不知道为什么，论文也没解释）。

#### 宽带模型的波束空间相参合成

宽带模型可以根据频域滤波转换为多个不同中心频率的窄带模型，可以按照波束空间转换的思路将不同窄带数据的协方差矩阵相残融合在一起，要求解释对于每一个频率的窄带模型，其各自所形成的$P$的波束响应器，在对应频率上，要求器波束响应要尽量一致。如何做到一致，以及如果不可能做到一致时，又会有怎样的取舍和近似，这一点有待研究，现在掌握的大概就行了。

在实践中，不同窄带模型的波束空间一致性不好，就好导致最后合成的效果不好。
