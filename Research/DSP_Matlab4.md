### 数字滤波

对于一个离散时间序列$x[n]$，其频谱为$X(e^{\mathrm{j}w})$，通常希望将其通过一个理想的滤波器$H_d(e^{\mathrm{j}w})$，使得输出的序列$y[n]$拥有希望的频谱特性（比如滤除某一特定频段）
$$
Y(e^{\mathrm{j}w}) = X(e^{\mathrm{j}w}) H_d(e^{\mathrm{j}w}) \\
y[n] = x[n] * h_d[n]
$$
然而，$h_d[n]$通常情况下是无限长且非因果的（比如$H_d(e^{\mathrm{j}w})$是理想低通滤波器），在时频域都无法完成（频域上$w$是连续得，无法积分，只能时域；然而时域上输入序列要与无限长的滤波器卷积）。所以希望找到一个特定长度$M$的序列$h[n]$，使得其$H(e^{\mathrm{j}w})$要尽量与$H_d(e^{\mathrm{j}w})$逼近。

#### 窗函数设计法

窗函数设计法通过直接加窗以截断无限长的$h_d[n]$序列，完成$H(e^{\mathrm{j}w})$对$H_d(e^{\mathrm{j}w})$的逼近。从理论可以证明证明，使用矩形窗进行阶段得到的近似$H(e^{\mathrm{j}w})$，在均方误差的逼近准则下是最优的。然而矩形窗虽然拥有最窄的过渡带，但会导致$H(e^{\mathrm{j}w})$具有Gibbs效应。通过选择其他窗函数，以展宽过度带的代价，可以缓解Gibbs效应。

#### 频率采样设计法

频率采样设计法通过对$H_d(e^{\mathrm{j}w})$进行采样，得到序列
$$
H[k] = H(e^{\mathrm{j} 2 \pi k / M}), \quad k = 0, \cdots , M - 1
$$
再用IDFT将$H[k]$转换为时域序列$h[n]$，而得到有限长的滤波器系数。可以知道，频域采样等效于时域的周期延拓，所以$h[n]$为$h_d[n]$以$M$点周期延拓后的主值序列
$$
h[n] = \sum\limits_{m = -\infty}^{\infty} h_d[n + mM] w_M[n]
$$
其中，$w_M[n]$为长度$M$的矩形窗。

为了评估$h[n]$对于理想滤波器的逼近程度，对其进行DTFT
$$
H(e^{\mathrm{j}w}) = \sum\limits_{m = 0}^{N - 1} H[m] \phi(w - 2 \pi m / N)
$$
其中，$\phi(w)$为内插函数（矩形窗的DTFT）
$$
\phi(w) = \frac{1}{N} \frac{\sin\left( wN / 2 \right)}{\sin\left(w / 2 \right)} e^{-\mathrm{j} w \frac{N - 1}{2}}
$$
所以，$H(e^{\mathrm{j}w})$就相当于对采样后的离散序列进行内插，以逼近$H_d(e^{\mathrm{j}w})$。

#### 频率采样设计法与DFT直接置零的关系

规范的对一个序列进行滤波操作，需要

- 滤波器设计，得到滤波器的有理传递函数$H(z)$；
- 滤波（时域线性卷积，频域相乘）。

然而，常有人进行以下操作

- 对序列进行DFT，得到离散频谱；
- 在将需要的离散频点置零，在IDTF转换为时域序列。

DFT直接置零来说相当于在离散频谱上直接乘上以$N$点（输入序列长度也是$N$）频域采样得到的$H[k]$。为了方便，我们先讨论$M = N$的频率采样法得到的输出序列频谱$Y(e^{\mathrm{j}w})$，由前可知
$$
\begin{aligned}
Y(e^{\mathrm{j}w}) &= X(e^{\mathrm{j}w}) H(e^{\mathrm{j}w}) \\
&= X(e^{\mathrm{j}w}) \sum\limits_{m = 0}^{N - 1} H[m] \phi(w - 2 \pi m / N)
\end{aligned}
$$
其可以视为先对$H[k]$进行差值后，在再频域上相乘。

对于DFT直接置零来说，输出序列的$N$点DFT为
$$
Y'[k] = X[k] H[k]
$$
其相当于时域$x[n]$与$h[n]$的$N$点循环卷积，其等价于线性卷积的输出序列$y[n]$的$N$点周期延拓
$$
y'[n] = \sum\limits_{m = -\infty}^{\infty} y[n + mN] w_M[n]
$$
由此，其DTFT为
$$
Y'(e^{\mathrm{j}w}) = \sum\limits_{k = 0}^{N - 1} Y[k] \phi(w - 2 \pi k / N)
$$
其中，$Y[k]$为对线性线性卷积输出序列频谱$Y(e^{\mathrm{j}w})$的$N$点离散采样
$$
\begin{aligned}
Y[k] &= Y(e^{\mathrm{j} 2 \pi k / N})  \\
&= X(e^{\mathrm{j}2 \pi k / N}) \sum\limits_{m = 0}^{N - 1} H[m] \phi(2 \pi k / N - 2 \pi m / N) \\
&= X[k] \sum\limits_{m = 0}^{N - 1} H[m] \phi(2 \pi (k - m) / N) \\
&= X[k] H[k]
\end{aligned}
$$
即采样点处直接相乘就可。

由此
$$
\begin{aligned}
Y'(e^{\mathrm{j}w}) &= \sum\limits_{k = 0}^{N - 1} Y[k] \phi(w - 2 \pi k / N) \\
&= \sum\limits_{k = 0}^{N - 1} X[k] H[k] \phi(w - 2 \pi k / N)

\end{aligned}
$$
其可以视为两个离散的频域相乘后，在进行差值。

#### 总结

频率采样设计法得到的滤波输出序列的频谱，相当于先对频域采样$H[k]$进行插值，再乘$X(e^{\mathrm{j}w})$。

而DFT直接置零相当于得到$X[k]H[k]$后，再进行插值。

也可以认为DFT直接置零相当于进行了频率采样设计法后，再在频率上进行$N$点采样的序列。由于$N$点频率采样设计法得到的序列长度为$2N - 1$，所以DFT置零的频谱是有损失的（时域发生了混叠）。