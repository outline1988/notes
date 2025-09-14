### 已知波形下的单源DOA最大似然估计

论文：Maximum likelihood angle estimation for signals with known waveforms

在白噪声背景下，我们假设DOA估计中的波形已知，但是带有一个位置的复振幅
$$
\begin{aligned}
\mathbf{x}(t) &= \mathbf{A} \mathbf{s}(t) + \mathbf{n}(t) \\
&= \mathbf{A} \mathbf{P}(t) \boldsymbol{\alpha} + \mathbf{n}(t)
\end{aligned}
$$
其中，我们知道信号$\mathbf{s}(t)$是已知波形的复振幅（每个信源的复振幅不同），所以信号与已知波形的关系为
$$
\begin{aligned}
\mathbf{s}(t) = \begin{bmatrix}
s_1(t) \\ \vdots \\ s_K(t)
\end{bmatrix} = 
\begin{bmatrix}
p_1(t) & & \\ & \ddots \\  & & p_K(t)
\end{bmatrix} \begin{bmatrix}
\alpha_1 \\ \vdots \\ \alpha_K
\end{bmatrix}
\end{aligned}
$$
其中，$p_k(t)$为已知波形。这种模型下，$\mathbf{P}(t)$的顺序只要给定，那么搜索到的结果$\theta_k$和$\alpha_k$就自动与对应波形$p_k(t)$匹配上了。

我们考虑一种最简单的情况，只存在一个源（EM算法中的M-step常常需要对单源模型进行估计，所以研究一个源的情况还是有价值的；同时，多源最大似然中间可以有近似简化化为多个单源的叠加），此时模型为
$$
\mathbf{x}(t) = \mathbf{a}(\theta) p(t) \alpha + \mathbf{n}(t)
$$
为了求解其最大似然估计MLE，需要对一下函数进行优化
$$
\begin{aligned}
q' &= \frac{1}{N}\sum\limits_{t = 1}^{N}\left[ \mathbf{x}(t) - \mathbf{a}(\theta) p(t) \alpha \right]^H\left[ \mathbf{x}(t) - \mathbf{a}(\theta) p(t) \alpha \right] \\
&= \frac{1}{N}\sum\limits_{t = 1}^{N}\left[\mathbf{x}^H(t) \mathbf{x}(t) - \mathbf{x}^H(t)\mathbf{a}(\theta) p(t) \alpha - \mathbf{a}^H(\theta)\mathbf{x}(t) p^*(t) \alpha^* + M \left| p(t) \alpha \right|^2 \right] \\
\end{aligned}
$$
抛弃无关项，并且展开向量，其中$z = [\mathbf{a}(\theta)]_2$即导向矢量的第二个元素，这里假设为均匀线阵
$$
\begin{aligned}
q &= \frac{1}{N}\sum\limits_{t = 1}^{N}\left[ - \mathbf{x}^H(t)\mathbf{a}(\theta) p(t) \alpha - \mathbf{a}^H(\theta)\mathbf{x}(t) p^*(t) \alpha^* + M \left| p(t) \alpha \right|^2 \right] \\
&=  -\frac{1}{N}\sum\limits_{t = 1}^{N} \left[ p(t) \alpha \sum\limits_{m = 1}^{M} x_m^*(t) z^{m - 1} + p^*(t) \alpha^* \sum\limits_{m = 1}^{M} x_m(t) z^{-(m - 1)} \right] + \left| \alpha \right|^2 \frac{M}{N}\sum\limits_{t = 1}^{N} \left|p(t)\right|^2 \\
&= q_1 + q_2
\end{aligned}
$$
其中，第一项为
$$
\begin{aligned}
q_1 &= -\frac{1}{N}\sum\limits_{t = 1}^{N} \left[ p(t) \alpha \sum\limits_{m = 1}^{M} x_m^*(t) z^{m - 1} + p^*(t) \alpha^* \sum\limits_{m = 1}^{M} x_m(t) z^{-(m - 1)} \right] \\
&= -\sum\limits_{m = 1}^{M} \left[ \frac{\alpha}{N} \sum\limits_{t = 1}^{N} p(t) x^*_m(t) z^{m - 1} + \frac{\alpha^*}{N} \sum\limits_{t = 1}^{N} p^*(t) x_m(t) z^{-(m - 1)} \right] \\
&= -\sum\limits_{m = 1}^{M} \left[ {\alpha} y^*_m z^{m - 1} + \alpha^* y_m z^{-(m - 1)} \right] \\
&= - \left[\alpha \mathbf{y}^H \mathbf{a}(\theta) + \alpha^* \mathbf{a}^H(\theta) \mathbf{y}\right] \\
&= -2 \operatorname{Re} \left[ \alpha^* \mathbf{a}^H(\theta) \mathbf{y} \right]
\end{aligned}
$$
其中，$\mathbf{y} = \frac{1}{N}\sum\limits_{i = 1}^{N} \mathbf{x}(t) p^*(t)$，表示将量测$\mathbf{x}(t)$的每一通道与$p(t)$分别作相关后，取得的峰值。故最终优化的目标函数为
$$
q = \left| \alpha \right|^2 \frac{M}{N} \mathbf{p}^H \mathbf{p} -2 \operatorname{Re} \left[ \alpha^* \mathbf{a}^H(\theta) \mathbf{y} \right]
$$
因为$q$还包含着$\alpha$项，所以目标函数分别对复振幅$\alpha = \bar{\alpha} + \tilde{\alpha}$的实部和虚部求导
$$
\begin{aligned}
\frac{\partial q}{\partial \bar{\alpha}} &= \frac{\partial }{\partial \bar{\alpha}}\left\{\left( \bar{\alpha}^2 + \tilde{\alpha}^2 \right) K - 2\operatorname{Re}\left[ (\bar{\alpha} - \mathrm{j} \tilde{\alpha}) \left(\bar{L} + \mathrm{j} \tilde{L}\right) \right]\right\} \\
&=\frac{\partial }{\partial \bar{\alpha}}\left\{\left( \bar{\alpha}^2 + \tilde{\alpha}^2 \right) K - 2 \left( \bar{\alpha}\bar{L} + \tilde{\alpha}\tilde{L} \right)\right\} \\
&= 2 \bar{\alpha}K - 2 \bar{L} = 0
\end{aligned}
$$
故，$\bar{\alpha} = \bar{L} / K$。同理，$\tilde{\alpha} = \tilde{L} / K$，所以复振幅可以表示为
$$
\alpha = \frac{L}{K} = \frac{N}{M}\frac{\mathbf{a}^H(\theta) \mathbf{y}}{ \mathbf{p}^H \mathbf{p}}
$$
这里实际上也可以不用对$\alpha$求导，因为在原式中，其为线性部分，所以可以用MLE中的线性参数分离的技巧，即若已知$\mathbf{a}(\theta) p(t)$，则此时$\alpha$可表示为
$$
\alpha = \left[ \sum\limits_{t = 1}^N \mathbf{a}^H p^*(t) \mathbf{a} p(t) \right]^{-1} \left[ \sum\limits_{t = 1}^{N} \mathbf{a}^H p^*(t) \mathbf{x}(t) \right] = \frac{N}{M}\frac{\mathbf{a}^H(\theta) \mathbf{y}}{ \mathbf{p}^H \mathbf{p}}
$$


最终将复振幅的表达式代入目标函数
$$
\begin{aligned}
q &= \left| \alpha \right|^2 \frac{M}{N} \mathbf{p}^H \mathbf{p} -2 \operatorname{Re} \left[ \alpha^* \mathbf{a}^H(\theta) \mathbf{y} \right] \\
&= \frac{N^2}{M^2}\frac{\left|\mathbf{a}^H(\theta) \mathbf{y}\right|^2}{ \| \mathbf{p}\|^4 } \frac{M}{N} \| \mathbf{p}\|^2 - 2 \operatorname{Re}\left[ \frac{N}{M}\frac{\left|\mathbf{a}^H(\theta) \mathbf{y}\right|^2}{ \| \mathbf{p}\|^2 } \right] \\
&= -\frac{N}{M}\frac{\left|\mathbf{a}^H(\theta) \mathbf{y}\right|^2}{ \| \mathbf{p}\|^2 }
\end{aligned}
$$
这个目标函数的物理过程可以视为，首先对每个阵元进行相关，得到一个最大值（匹配滤波的峰值），由此可以认为得到了一个单快拍的数据，对这个单快拍的数据进行一维的搜索（periodgram），从而得到DOA的估计，最终再由DOA估计得到复振幅的估计值。

**复增益的相位是否对估计精度的影响**

如果$\alpha$的相位未知，那么最后DOA的优化目标就是
$$
q_1 = - \left| \mathbf{a}^H(\theta) \mathbf{y} \right|
$$
其引入更多的噪声项，因为噪声项也取模了，噪声的能量也集中了。

若$\alpha$的相位已知，则可以用这个已知的相位来补偿目标函数，使得能量集中在实部，DOA的优化目标为
$$
q_1 = -\operatorname{Re} \left[\mathbf{a}^H(\theta) \mathbf{y} \right]
$$
此时常数的表达式为
$$
\alpha = \frac{N}{M} \frac{\operatorname{Re}(\mathbf{a}^H(\theta)\mathbf{y})}{\mathbf{p}^H \mathbf{p}}
$$
一是DTFT后取实部，二是DTFT后取模值，前者对应复增益的相位已知，后者对应复增益的相位未知。如果复增益相位已知，那么我能够直接使用已知的相位在一阶的时候就把能量集中在实部；而相位未知时，只能取模，而取模会使得噪声的影响变得更大，从某种程度上信噪比降低了，所以相位未知的情况下估计精度会更差。

### 已知波形下的DOA最大似然估计（单源情况）

前面直接从单源的角度出发，求出了最大似然估计，现在从一般的情况出发，我们需要解决如下的优化问题
$$
\begin{aligned}
J &= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{B}(t) \boldsymbol{\alpha} \right\|^2
\end{aligned}
$$
其中，$\mathbf{B}(t) = \mathbf{A}(\boldsymbol{\theta})\mathbf{P}(t)$，将在后面简写为$\mathbf{B}$。可以知道，若已知DOA，则$\boldsymbol{\alpha}$的最大似然估计为（详见Matrix_Notes1）
$$
\boldsymbol{\alpha} = \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right)
$$
则代入原式
$$
\begin{aligned}
J = \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{B} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right) \right\|^2
\end{aligned}
$$
我们假设单源的情况，此时
$$
\mathbf{B}(t) = \mathbf{a}({\theta}) p(t)
$$
则
$$
\begin{aligned}
\sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} &= \sum\limits_{t = 1}^{N} \mathbf{a}^H({\theta}) p^*(t) \mathbf{a}({\theta}) p(t) \\
&= M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2
\end{aligned}
$$

$$
\begin{aligned}
\sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) &= \sum\limits_{t = 1}^{N}\mathbf{a}^H({\theta}) p^*(t) \mathbf{x}(t) \\
&= \mathbf{a}^H({\theta}) \sum\limits_{t = 1}^{N} p^*(t) \mathbf{x}(t) \\
&= \mathbf{a}^H({\theta}) \mathbf{y}
\end{aligned}
$$

其中，$\mathbf{y} = \sum_{t = 1}^{N} p^*(t) \mathbf{x}(t)$代表将已知的波形对阵元每个通道的接收数据做一次相关得到的一个列向量，列向量的每个元素可以视为每个通道的接收数据做匹配滤波的输出。

由此，代入原式
$$
\begin{aligned}
J &= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{B} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right) \right\|^2 \\
&= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{a}({\theta}) p(t) \frac{\mathbf{a}^H({\theta}) \mathbf{y}}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2}  \right\|^2 \\
&= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) -  p(t) \mathbf{Q} \mathbf{y} \right\|^2 \\
\end{aligned}
$$
其中
$$
\mathbf{Q} = \frac{\mathbf{a}(\theta) \mathbf{a}^H(\theta)}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2} = L \mathbf{a}\mathbf{a}^H
$$
由此
$$
\begin{aligned}
J &= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) -  p(t) \mathbf{Q} \mathbf{y} \right\|^2 \\
&= \sum\limits_{t = 1}^{N} \left[  \mathbf{x}(t) -  p(t) \mathbf{Q} \mathbf{y} \right]^H \left[  \mathbf{x}(t) -  p(t) \mathbf{Q} \mathbf{y} \right]\\
&= \sum\limits_{t = 1}^{N} \left[ \mathbf{x}^H(t)\mathbf{x}(t) - \mathbf{x}^H(t) p(t) \mathbf{Q}\mathbf{y} - p^*(t) \mathbf{y}^H \mathbf{Q}^H \mathbf{x}(t) + |p(t)|^2 \mathbf{y}^H\mathbf{Q}^H\mathbf{Q}\mathbf{y} \right]
\end{aligned}
$$
抛弃无关项
$$
\begin{aligned}
J' &= \sum\limits_{t = 1}^{N} \left[  - \mathbf{x}^H(t) p(t) \mathbf{Q}\mathbf{y} - p^*(t) \mathbf{y}^H \mathbf{Q}^H \mathbf{x}(t) + |p(t)|^2 \mathbf{y}^H\mathbf{Q}^H\mathbf{Q}\mathbf{y} \right] \\
&= -J_1 + J_2
\end{aligned}
$$
其中
$$
\begin{aligned}
J_2 &= \sum\limits_{t = 1}^{N} |p(t)|^2 \mathbf{y}^H\mathbf{Q}^H\mathbf{Q}\mathbf{y} \\
&= \sum\limits_{t = 1}^{N} |p(t)|^2 \mathbf{y}^H L^2 \mathbf{a}\mathbf{a}^H \mathbf{a}\mathbf{a}^H \mathbf{y} \\
&= \frac{|\mathbf{a}^H \mathbf{y}|^2}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2}
\end{aligned}
$$

$$
\begin{aligned}
J_1 &= \sum\limits_{t = 1}^{N}  \left[\mathbf{x}^H(t) p(t) \mathbf{Q}\mathbf{y} + p^*(t) \mathbf{y}^H \mathbf{Q}^H \mathbf{x}(t)\right] \\
&= 2 \operatorname{Re} \left[ \sum\limits_{t = 1}^{N} p(t) \mathbf{x}^H(t) \mathbf{Q}\mathbf{y} \right] \\
&= \frac{2 |\mathbf{a}^H \mathbf{y}|^2}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2}
\end{aligned}
$$

故最终
$$
J' = \frac{|\mathbf{a}^H(\theta) \mathbf{y}|^2}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2}
$$
为了最大化这个目标函数，只需要对每个通道使用已知的信号波形做相关（匹配滤波的峰值），再将所有阵元的相关结果做一次一维搜索（DTFT）即可，所以在单源已知波形的情况下，最大似然估计直接先包含了匹配滤波之后，再做DTFT。

在获得了DOA的估计值后，复振幅的估计值为
$$
\begin{aligned}
\boldsymbol{\alpha} &= \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right) \\
&= \frac{\mathbf{a}^H(\theta) \mathbf{y}}{M \sum\limits_{t = 1}^{N} \left|p(t)\right|^2}
\end{aligned}
$$
这个结果也与最开始同论文一样推导方式的结果是一样的。

### 已知波形下多源但可分辨的最大似然估计

从这里开始
$$
\begin{aligned}
J = \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{B} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right) \right\|^2
\end{aligned}
$$
其中
$$
\mathbf{B}(t) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{P}(t)
$$
则（假设各个源之间DOA相隔较远，所以对应导向矢量相互正交）
$$
\begin{aligned}
\sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} &= \mathbf{P}^H \mathbf{A}^H\mathbf{A} \mathbf{P} \\
&= M \begin{bmatrix}
\sum\limits_{t = 1}^{N} \left|p_1(t)\right|^2 & & \\
& \ddots & \\
& & \sum\limits_{t = 1}^{N} \left|p_K(t)\right|^2
\end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
\sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) &= \sum\limits_{t = 1}^{N}\mathbf{P}^H \mathbf{A}^H \mathbf{x}(t) \\
&=\begin{bmatrix}
\sum\limits_{t = 1}^{N} p_1^*(t) \mathbf{a}^H(\theta_1) \mathbf{x}(t) \\
\vdots \\
\sum\limits_{t = 1}^{N} p_K^*(t) \mathbf{a}^H(\theta_K) \mathbf{x}(t)
\end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{a}^H (\theta_1) \mathbf{y}_1 \\
\vdots \\
\mathbf{a}^H (\theta_K) \mathbf{y}_K
\end{bmatrix}
\end{aligned}
$$

其中
$$
\mathbf{y}_k = \sum\limits_{t = 1}^{N} p_k^*(t) \mathbf{x}(t)
$$
表示使用某个波形$p_k(t)$来对多源叠加的观测$\mathbf{x}(t)$的每一个通道进行相关后取峰值。故原式可再次写为
$$
\begin{aligned}
J &= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - \mathbf{B} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{B} \right)^{-1} \left( \sum\limits_{t = 1}^{N} \mathbf{B}^H \mathbf{x}(t) \right) \right\|^2 \\
&= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) - 
\begin{bmatrix}
\mathbf{a}(\theta_1) & \cdots & \mathbf{a}(\theta_K)
\end{bmatrix}
\begin{bmatrix}
p_1(t) & & \\
& \ddots & \\
& & p_K(t)
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sum\limits_{t = 1}^{N} \left|p_1(t)\right|^2} & & \\
& \ddots & \\
& & \frac{1}{\sum\limits_{t = 1}^{N} \left|p_K(t)\right|^2}
\end{bmatrix} \begin{bmatrix}
\mathbf{a}^H (\theta_1) \mathbf{y}_1 \\
\vdots \\
\mathbf{a}^H (\theta_1) \mathbf{y}_K
\end{bmatrix}  \right\|^2 \\
&= \left\| \mathbf{x}(t) - \begin{bmatrix}
\mathbf{a}(\theta_1) & \cdots & \mathbf{a}(\theta_K)
\end{bmatrix} \begin{bmatrix}
\frac{p_1(t) }{\sum\limits_{t = 1}^{N}  \left|p_1(t)\right|^2}\mathbf{a}^H (\theta_1) \mathbf{y}_1 \\
\vdots \\
\frac{p_K(t) }{\sum\limits_{t = 1}^{N}  \left|p_K(t)\right|^2}\mathbf{a}^H (\theta_K) \mathbf{y}_K
\end{bmatrix} \right\| \\
&= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) -  \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right\|^2 \\
\end{aligned}
$$
其中
$$
\mathbf{Q}_k = \frac{\mathbf{a}(\theta_k) \mathbf{a}^H(\theta_k)}{M \sum\limits_{t = 1}^{N} \left|p_k(t)\right|^2}
$$
由此
$$
\begin{aligned}
J &= \sum\limits_{t = 1}^{N} \left\|  \mathbf{x}(t) -  \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right\|^2 \\
&= \sum\limits_{t = 1}^{N} \left[  \mathbf{x}(t) -  \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right]^H \left[  \mathbf{x}(t) -  \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right]\\
&= \sum\limits_{t = 1}^{N} \left[ \mathbf{x}^H(t)\mathbf{x}(t) - \mathbf{x}^H(t) \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k - \sum\limits_{k = 1}^{N}p_k(t) \mathbf{y}_k^H \mathbf{Q}_k^H  \mathbf{x}(t) + |p(t)|^2 \mathbf{y}^H\mathbf{Q}^H\mathbf{Q}\mathbf{y} \right]
\end{aligned}
$$


抛弃无关项
$$
\begin{aligned}
J' &= \sum\limits_{t = 1}^{N} \left[  - \mathbf{x}^H(t) \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k - \sum\limits_{k = 1}^{N}p_k(t) \mathbf{y}_k^H \mathbf{Q}_k^H  \mathbf{x}(t) + \left\| \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right\|^2 \right] \\
&= -J_1 + J_2
\end{aligned}
$$
其中
$$
\begin{aligned}
J_2 &= \sum\limits_{t = 1}^{N} \left\| \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right\|^2 \\
&= \sum\limits_{t = 1}^{N}\left[ \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right]^H\left[ \sum\limits_{k = 1}^{N}p_k(t) \mathbf{Q}_k \mathbf{y}_k \right] \\
&= \sum\limits_{k = 1}^{K}  \sum\limits_{t = 1}^{N} |p(t)|^2 \mathbf{y}^H \mathbf{Q}_k^H \mathbf{Q}_k \mathbf{y} \\
&= \sum\limits_{k = 1}^{K}  \frac{|\mathbf{a}^H(\theta_k) \mathbf{y}_k|^2}{M \sum\limits_{t = 1}^{N} \left|p_k(t)\right|^2}
\end{aligned}
$$


$J_1$项应该也是类似的结果。最终，需要最大化以下目标函数
$$
J = \sum\limits_{k = 1}^{K}  \frac{|\mathbf{a}^H(\theta_k) \mathbf{y}_k|^2}{M \sum\limits_{t = 1}^{N} \left|p_k(t)\right|^2}
$$
可以看到，对于每一个$k$，我们首先需要将所有通道做匹配滤波，然后再将匹配滤波的峰值输出做一维的搜索（DTFT）。

实际上，从这里的推导来看，我们得到了一个在信源分隔较大情况下关于已知信号波形DOA估计的一个最大似然解。

从物理意义上来看，使用特定信号先做相关相当于提取出观测数据中想要的某个特定信号，而抑制其他的信号，由此再在这个信号上做DOA估计。如果从这一角度来看待的化，实际上，如果假设入射的信源彼此之间不相关，那么即使是相隔很近的信源，最终也能分隔开，这也是DEML这篇文章中所讨论的：Computationally efficient angle estimation for signals with known waveforms。虽然这篇文章没有显式提到相关的概念，但是其推导的方法隐含这做了相关，并且这篇论文是在假设信号不相关情况下的大样本最大似然近似方法，所以文中提到的种种性质从相关的角度来说很容易理解。































