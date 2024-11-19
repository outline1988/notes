## 随机性已知假设

随机性已知假设就是假设信号为已知的WSS随机过程，即知道其均值和协方差矩阵（功率谱密度）。

### TDOA单信号源单接收通道

信号模型如下
$$
x(t) = s(t - \tau) + n(t)
$$
其中，$s(t)$与$n(t)$是独立的零均值WSS高斯随机过程，且已知其协方差矩阵（功率谱）。

由于$s(t - \tau)$为平稳过程，所以其特性不随时间平移的改变而改变，所以即使平移了一个版本，也无法得到其关于时延的相关信息。也即，只从一个接收通道来分析，延迟版本的WSS随机过程的统计特性不会发生任何改变，所以无法估计时延。

定量来说，$s(t - \tau)$的相关为
$$
\begin{aligned}
E\left[ s(t_1 - \tau) s^*(t_2 - \tau)  \right] &= r_{ss}(t_1 - \tau - (t_2 - \tau)) \\
&= r_{ss}(t_1 - t_2)
\end{aligned}
$$
与延迟$\tau$无关，所以无法得到关于时延的信息，随机性假设的单通道时延估计无法完成。

从物理意义上来看，一个WSS高斯随机过程只用了两个与时间无关的参数均值和协方差就能描述，故无论什么WSS随机过程，其延迟必然也能视为同一个WSS随机信号的另一种可能的实现，所以不能提取出关于时延的信息。必须有另一个通道的数据作为参照，表明本次WSS随机过程的实现，从而比对出时延信息。

### TDOA单信号源双接收通道

论文：On the Cramer- Rao bound for time delay and Doppler estimation

同样是时延估计问题，本次不再拥有已知的确定性信号，而是拥有已知的WSS高斯随机信号，通过两个传感器接收到的观测信号来估计TDOA，模型如下（实）
$$
\begin{aligned}
x_1(t) &= s(t) + n_1(t) \\
x_2(t) &= s(t - \tau) + n_2(t)
\end{aligned}
$$
其中，$s(t)$、$n_1(t)$和$n_2(t)$为互相独立的WSS高斯随机过程，分别有功率谱$P_{ss}(f)$、$P_{n_1n_1}(f)$和$P_{n_2n_2}(f)$，由此可以写出
$$
P_{x_1x_1}(f) = P_{ss}(f) + P_{n_1n_1}(f) \\
P_{x_2x_2}(f) = P_{ss}(f) + P_{n_2n_2}(f) \\
$$
同时
$$
\begin{aligned}
E\left[ x_1(t_1) x_2(t_2) \right] &= E\left[ \left(s(t_1) + n_1(t_1)\right) (s(t_2 - \tau) + n_2(t_2)) \right] \\
&= E\left[ s(t_1) s(t_2 - \tau) \right] \\
&= r_{ss}\left((t_1 - t_2) + \tau\right)
\end{aligned}
$$
由此
$$
P_{x_1x_2}(f; \tau)  = P_{ss}(f) \exp(\mathrm{j} 2 \pi f \tau)
$$
将功率谱写成矩阵形式
$$
\mathbf{P}(f; \tau) = \begin{bmatrix}
P_{ss}(f) + P_{n_1n_1}(f) & P_{ss}(f) \exp(\mathrm{j} 2 \pi f \tau)\\
P_{ss}(f) \exp(-\mathrm{j} 2 \pi f \tau)& P_{ss}(f) + P_{n_2n_2}(f)
\end{bmatrix} \\
$$
由此FIM可计算为
$$
\begin{aligned}
{I}(\tau) &= \frac{T}{2} \int_{- \infty}^{\infty} \operatorname{tr}\left\{ \mathbf{P}^{-1}(f; \tau) \frac{\partial \mathbf{P}(f; \tau)}{\partial\tau}\mathbf{P}^{-1}(f; \tau) \frac{\partial \mathbf{P}(f; \tau)}{\partial\tau} \right\} \mathrm{d} f \\
&= \frac{T}{2} \int_{- \infty}^{\infty} \operatorname{tr}\left\{ 
\begin{bmatrix}
(2 \pi f)^2 P^2_{ss}(f) & x \\
x & (2 \pi f)^2P^2_{ss}(f) \\
\end{bmatrix} \frac{1}{\det \left[\mathbf{P}(f; \tau) \right]}
\right\} \mathrm{d} f
\end{aligned}
$$
其中，$x$代表无关紧要的数值
$$
\det \left[\mathbf{P}(f; \tau) \right] = \left( P_{ss}(f) + P_{n_1n_1}(f) \right) \left( P_{ss}(f) + P_{n_1n_1}(f) \right) - P^2_{ss}(f)
$$
由此
$$
{I}(\tau) = T \int_{-\infty}^{\infty} \frac{(2 \pi f)^2 \frac{P_{ss}(f)}{P_{n_1n_1(f)}}\frac{P_{ss}(f)}{P_{n_2n_2(f)}}}{1 + \frac{P_{ss}(f)}{P_{n_1n_1(f)}} + \frac{P_{ss}(f)}{P_{n_2n_2(f)}}} \mathrm{d} f
$$

从某种程度上，WSS随机信号时延$\tau$估计的Fisher信息取决于随机信号功率谱的有效带宽。比如说最为典型的随机相位单频信号来说，直观上其双通道时延精度是很差的，因为两个单频信号的差别只在于相位差，而这个相位差又由于随机相位被模糊了，从其谱宽也能侧面印证这一点。对于某个功率谱稍宽的随机信号，在时域上可以想象为很多个不同频率的随机相位单频信号的叠加，不同的频率相当于进行了解模糊，而越多的频率就提供了更强大的解模糊能力，所以时延精度高。

**论文中同样还提到了多信号源双接收通道以及信号功率谱未知的情况，但没细看。**



### TDOA双径单接收通道

论文：On the Cramer- Rao bound for time delay and Doppler estimation

在双径单接收通道场景下，信号模型（实）为
$$
x(t) = s(t) + \alpha s(t - \tau) + n(t)
$$
其中，$s(t)$和$n(t)$是已知的独立零均值WSS高斯随机过程，$\alpha$为已知的衰减因子，待估计参数为$\tau$，求其CRB。

可知$x(t)$为零均值WSS高斯随机过程，且功率谱密度包含了参数$\tau$，可以使用零均值WSS渐近FIM公式来计算。

首先计算出$x(t)$的相关矩阵
$$
\begin{aligned}
E\left[ x(t_1) x(t_2 - \tau) \right] &= E\left[ \left(s(t_1) + \alpha s(t_1 - \tau) + n(t_1)\right) \left(s(t_2) + \alpha s(t_2 - \tau) + n(t_2)\right) \right] \\
&= E\left[ s(t_1)s(t_2) + \alpha^2 s(t_1 - \tau)s(t_2 - \tau) + \alpha s(t_1 - \tau)s(t_2) + \alpha s(t_1)s(t_2 - \tau) + n(t_1)n(t_2) \right] \\
&= r_{ss}(t_1 - t_2) + \alpha^2 r_{ss}(t_1 - t_2) + \alpha r_{ss}(t_1 - t_2 - \tau) + \alpha r_{ss}(t_1 - t_2 + \tau) + r_{nn}(t_1 - t_2)
\end{aligned}
$$
故$x(t)$的功率谱密度为
$$
\begin{aligned}
P_{xx}(f; \tau) &= P_{ss}(f) + \alpha P_{ss}(f) \exp(-\mathrm{j} 2 \pi f \tau) + \alpha P_{ss}(f) \exp(\mathrm{j} 2 \pi f \tau) + P_{nn}(f) \\
&= [1 + \alpha^2 + 2 \alpha \cos(2 \pi f \tau)] P_{ss}(f) + P_{nn}(f)
\end{aligned}
$$
所以带入公式
$$
\begin{aligned}
I(\tau) &= \frac{T}{2} \int_{-\infty}^{\infty} \frac{\partial \ln P_{xx}(f; \tau)}{\partial \tau} \frac{\partial \ln P_{xx}(f; \tau)}{\partial \tau} \mathrm{d} f \\
&= \frac{T}{2} \int_{-\infty}^{\infty} \left[\frac{1}{P_{xx}(f; \tau)} \frac{\partial P_{xx}(f; \tau) }{\partial \tau}\right]^2 \mathrm{d} f \\
&= \frac{T}{2} \int_{-\infty}^{\infty} \frac{P_{ss}^2(f) 4 \alpha^2\sin^2(2 \pi f \tau) 4 \pi^2 f^2  }{P_{xx}^2(f; \tau)} \mathrm{d} f \\
&= 8 \alpha^2 \pi^2 T \int_{-\infty}^{\infty} \frac{f^2 P_{ss}^2(f) \sin^2(2 \pi f \tau)   }{P_{xx}^2(f; \tau)} \mathrm{d} f \\
\end{aligned}
$$
论文中还讨论了接下来的一些分析，这里就暂时讨论到这。

### FDOA单信号源单接收通道

论文：On the Cramer- Rao bound for time delay and Doppler estimation

与论文中的信号模型略微不同（复）
$$
x(t) = s(t) \exp(\mathrm{j} 2 \pi f_d t) + n(t)
$$
其中，$s(t)$和$n(t)$是已知的独立零均值WSS高斯随机过程，$f_d$为待估计参数。可知$x(t)$是WSS零均值高斯随机过程，故使用复数的WSS渐近FIM公式。

首先计算$P_{xx}(f; f_d)$
$$
\begin{aligned}
r_{xx}(t_1 - t_2) &= E\left[ s(t_1) \exp(\mathrm{j} 2 \pi f_d t) s(t_2) \exp(-\mathrm{j} 2 \pi f_d t) \right] \\
&= r_{ss}(t_1 - t_2) \exp\left(\mathrm{j} 2 \pi f_d (t_1 - t_2)\right)
\end{aligned}
$$
由此
$$
P_{xx}(f; f_d) = P_{ss}(f - f_d) + P_{nn}(f)
$$
带入公式
$$
\begin{aligned}
I(f_d) &= {T} \int_{-\infty}^{\infty} \frac{\partial \ln P_{xx}(f; f_d)}{\partial f_d} \frac{\partial \ln P_{xx}(f; f_d)}{\partial f_d} \mathrm{d} f \\
&= {T} \int_{-\infty}^{\infty} \left[\frac{1}{P_{xx}(f; f_d)} \frac{\partial P_{xx}(f; f_d) }{\partial f_d}\right]^2 \mathrm{d} f \\
&= {T} \int_{-\infty}^{\infty} \frac{1}{P_{xx}^2(f; f_d) } \left[\frac{\partial P_{ss}(f - f_d) }{\partial f_d}\right]^2\mathrm{d} f \\
&= {T} \int_{-\infty}^{\infty}  \left[\frac{1}{P_{xx}(f + f_d; f_d) } \frac{\partial P_{ss}(f) }{\partial f}\right]^2\mathrm{d} f \\
&= {T} \int_{-\infty}^{\infty}  \left[\frac{P_{ss}(f)}{P_{ss}(f) + P_{nn}(f + f_d) } \frac{1}{P_{ss}(f)} \frac{\partial P_{ss}(f) }{\partial f}\right]^2\mathrm{d} f \\
&= {T} \int_{-\infty}^{\infty}  \left[\frac{P_{ss}(f) / P_{nn}(f + f_d)}{1 + P_{ss}(f)/P_{nn}(f + f_d) } \frac{\partial \ln P_{ss}(f) }{\partial f}\right]^2\mathrm{d} f \\
&\approx {T} \int_{-\infty}^{\infty}  \left[\frac{P_{ss}(f) / P_{nn}(f)}{1 + P_{ss}(f)/P_{nn}(f) } \frac{\partial \ln P_{ss}(f) }{\partial f}\right]^2\mathrm{d} f \\
\end{aligned}
$$
从某种程度上，$f_d$的Fisher信息主要取决于信号功率谱密度的尖锐度，因为此时没有另一个通道作为参考，所以本次随机过程的知识只能从功率谱密度来获得，所以需要依赖于功率谱密度的尖锐程度来确定。对于有另一个接收通道的FDOA估计来说，由于有一个不带多普勒调制信号可以作为本次随机过程实现的参考，所以实际上与功率谱的形状没有关系了，只依赖于信噪比和时长。

论文中假设的信号模型为
$$
x(t) = s(\beta t)  + n(t)
$$
其中，$\beta = 1 + v / c$。又由于$f_d = 2 v / \lambda$，所以$\beta$和$f_d$呈线性关系，所以FIM也只差一个线性系数。

### FDOA单信号源双接收通道

这里给出一个计算单独FDOA情况下关于$f_d$的计算思路。对于TDOA的单源双通道来说，由于假设$s[n]$为WSS随机过程，所以可以直接带入公式计算。对于FDOA的模型来说，从频域的视角来看，其类似于时延估计的模型
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n] \exp(\mathrm{j}2\pi f_d n) + w_2[n] \\
\end{aligned}
$$
转为频域
$$
\begin{aligned}
&X_1[k] = S[k] + W_1[n] \\
&X_2[k] = S[k - d] + W_2[n]
\end{aligned}
$$
其在频域中表现为某个函数的平移。单独对$S[K]$进行讨论，其满足零均值独立高斯过程，但每个频率的方差不同，导致了其不是WSS平稳过程，不能使用WSS渐近FIM公式。

不过总归来说，从时延的角度来思考这个问题，有更多的思考。比如对于单通道时延估计来说，因为信号是平稳的，所以单通道的延迟不能提供任何时延估计的信息。但是对于单通道频延估计来说，其信号非平稳而独立，能够提供关于时延的信息，所以能够进行估计；再比如，WSS信号的时延估计精度是取决于其功率谱的有效带宽的，也即信号所占频带越大，理论精度就好。

### TDOA/FDOA单信号源双接收通道

论文：On the Cramer- Rao bound for time delay and Doppler estimation. 

The joint estimation of differential delay, Doppler, and phase. 

联合估计TDOA/FDOA的模型如下（还增加了相位）
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n  - n_0] \exp(\mathrm{j}2\pi f_d n) \exp(\mathrm{j} \phi) + w_2[n] \\
\end{aligned}
$$
经过一大堆我不知道怎么做的推导后，可以得到参数$\boldsymbol\xi = [\phi, \tau_0, w_d]^T$的FIM为
$$
\mathbf{I}(\boldsymbol\xi) = T\begin{bmatrix}
\int_{-\infty}^{\infty}\psi(f) \mathrm{d} f & \int_{-\infty}^{\infty}(2\pi f)\psi(f) \mathrm{d} f & 0 \\
\int_{-\infty}^{\infty}(2\pi f)\psi(f) \mathrm{d} f & \int_{-\infty}^{\infty}(2 \pi f)^2\psi(f) \mathrm{d} f & 0 \\
0 & 0 & \frac{T^2}{12}\int_{-\infty}^{\infty}\psi(f) \mathrm{d} f

\end{bmatrix}
$$
其中
$$
\begin{aligned}
\psi(f)  &= \frac{\left|P_{x_1x_2}(f)\right|^2}{P_{x_1x_1}(f) P_{x_2x_2}(f) - \left|P_{x_1x_2}(f)\right|^2} \\
&= \frac{P_{ss}^2(f)}{\left[P_{ss}(f) + P_{w_1w_1}(f)\right]\left[P_{ss}(f) + P_{w_2w_2}(f)\right] - P_{ss}^2(f)}
\end{aligned}
$$
从某种程度上可以将其视其为信噪比的功率密度。


## 确定性已知假设

### TDOA单信号源单接收通道

论文：Frequency domain Cramer-Rao bound for Gaussian processes

类似于Kay书中P39中的已知的确知信号中时延参数的检测，增加了一个衰减常数
$$
x(t) = \alpha s(t - \tau) + n(t)
$$
其中，$s(t)$为已知的确定实信号，$n(t)$为拥有已知功率谱密度$P_{nn}(f)$的WSS高斯随机过程。要确定的参数有$\boldsymbol\theta = [\alpha , \tau]^T$，确定该模型的FIM。

由于该问题为一般均值的WSS高斯随机过程的FIM计算问题，所以可以直接使用上面的公式
$$
\begin{aligned}
d_1 (f; \boldsymbol\theta) &= \frac{1}{\sqrt{T}} \operatorname{FT}\left\{\frac{\partial }{\partial \alpha } \left[ \alpha s(t - \tau) \right] \right\} \\
&= \frac{1}{\sqrt{T}} S(f) \exp(-\mathrm{j} 2 \pi f \tau) 
\end{aligned}
$$

$$
\begin{aligned}
d_2 (f; \boldsymbol\theta) &= \frac{1}{\sqrt{T}} \operatorname{FT}\left\{\frac{\partial }{\partial \tau } \left[ \alpha s(t - \tau) \right] \right\} \\
&= \frac{1}{\sqrt{T}} \operatorname{FT}\left\{-\alpha \frac{\partial s(t') }{\partial t'}
\bigg|_{t' = t - \tau} \right\} \\
&= -\frac{1}{\sqrt{T}} \mathrm{j} 2 \pi f \alpha S(f) \exp(-\mathrm{j} 2 \pi f \tau)
\end{aligned}
$$

所以可以得到
$$
\begin{aligned}
\left[\mathbf{I}(\boldsymbol\theta) \right]_{11} &= T\int_{- \infty}^{\infty} \frac{{d}_1^*(f;  \boldsymbol{\theta}) {d}_1(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} \mathrm{d} f \\
&= \int_{-\infty}^{\infty} \frac{ |S(f)|^2}{P_{nn}(f; \boldsymbol\theta)}\mathrm{d} f
\end{aligned}
$$

$$
\begin{aligned}
\left[\mathbf{I}(\boldsymbol\theta) \right]_{22} &= T\int_{- \infty}^{\infty} \frac{{d}_2^*(f;  \boldsymbol{\theta}) {d}_2(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} \mathrm{d} f \\
&= (2 \pi \alpha)^2 \int_{-\infty}^{\infty} \frac{f^2 |S(f)|^2}{P_{nn}(f; \boldsymbol\theta)}\mathrm{d} f
\end{aligned}
$$

$$
\begin{aligned}
\left[\mathbf{I}(\boldsymbol\theta) \right]_{12} &= T\int_{- \infty}^{\infty} \frac{{d}_1^*(f;  \boldsymbol{\theta}) {d}_2(f; \boldsymbol{\theta})}{P_{nn}(f; \boldsymbol{\theta})} \mathrm{d} f \\
&= -\mathrm{j}2 \pi \alpha \int_{-\infty}^{\infty} \frac{f |S(f)|^2}{P_{nn}(f; \boldsymbol\theta)}\mathrm{d} f \\
&= 0
\end{aligned}
$$

虽然这里增加了衰减因子这个参数，但是不影响作为时延本身的FIM。





