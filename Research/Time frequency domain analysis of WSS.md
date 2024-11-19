## WSS随机矢量的协方差矩阵在时域和频域的分析

### TDOA时域协方差分析

进行TDOA估计的信号模型如下
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n - n_0] + w_2[n]
\end{aligned}
$$
其中，$s[n]$、$w_1[n]$和$w_2[n]$都是WSS随机列矢量$\mathbf{x}$、$\mathbf{w}_1$和$\mathbf{w}_2$中的元素，且相互独立，分别拥有相关函数$r_{ss}[n]$、$r_{w_1w_1}[n]$和$r_{w_2 w_2}[n]$，则可以分别得到自协方差矩阵和互协方差矩阵，首先是$\mathbf{x}_1$的自协方差矩阵
$$
\begin{aligned}
\mathbf{C}_{x_1x_1} &= E\left[ (\mathbf{s} + \mathbf{w}_1) (\mathbf{s} + \mathbf{w}_1)^H \right]\\
&= \mathbf{C}_{ss} + \mathbf{C}_{w_1w_1}
\end{aligned}
$$

对于$\mathbf{x}_2$的自协方差矩阵
$$
\begin{aligned}
\left[ \mathbf{C}_{x_2x_2} \right]_{nm} &= E\left[ (s[n - n_0] + w_2[n]) (s[m - n_0] + w_2[m])^* \right] \\
&= E\left[ s[n - n_0]s^*[m - n_0] \right] + E\left[ w_2[n - n_0]w_2^*[m - n_0] \right] \\
&= r_{ss}[n - m] + r_{w_1w_1}[n - m]
\end{aligned}
$$
所以
$$
\mathbf{C}_{x_2x_2} = \mathbf{C}_{x_1 x_1}
$$
直观上来说，因为WSS随机过程本身的特性就是前二阶矩随时间不会发生改变，所以WSS随机过程就算进行了一个延迟，也不会改变其本身的自相关特性，所以无论延迟与否，自协方差矩阵都是相同的。

对于$\mathbf{x}_1$与$\mathbf{x}_2$的互协方差矩阵来说
$$
\begin{aligned}
\left[ \mathbf{C}_{x_1x_2} \right]_{nm} &= E\left[ (s[n] + w_1[n]) (s[m - n_0] + w_2[m])^* \right] \\
&= E\left[ s[n]s^*[m - n_0] \right] \\
&= r_{ss}[n - m + n_0]
\end{aligned}
$$
所以$\mathbf{C}_{x_1x_2}$有这样的形状
$$
\mathbf{C}_{x_1x_2} = \begin{bmatrix}
r_{ss}[n_0] & r_{ss}[n_0 - 1] & \cdots & r_{ss}[n_0 - N + 1] \\
r_{ss}[n_0 + 1] & r_{ss}[n_0] & \cdots & r_{ss}[n_0 - N] \\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[n_0 + N - 1] & r_{ss}[n_0 + N - 2] & \cdots & r_{ss}[0]
\end{bmatrix}
$$
相比于$\mathbf{C}_{ss}$本身的形状来说
$$
\mathbf{C}_{ss} = \begin{bmatrix}
r_{ss}[0] & r_{ss}[ - 1] & \cdots & r_{ss}[ - N + 1] \\
r_{ss}[  1] & r_{ss}[0] & \cdots & r_{ss}[ - N] \\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[  N - 1] & r_{ss}[  N - 2] & \cdots & r_{ss}[0]
\end{bmatrix}
$$
$\mathbf{C}_{x_1x_2}$好像$\mathbf{C}_{ss}$向右平移了$n_0$的单位，并且使得所有对角线上新增加的值保持统一，由此仍然保持Toeplitz性质，使得$\mathbf{x}_1$与$\mathbf{x}_2$互为平稳。

### FDOA时域协方差分析

FDOA的模型为
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n] \exp(\mathrm{j} 2 \pi f_d n) + w_2[n]
\end{aligned}
$$
$\mathbf{C}_{x_1x_1}$的与前面的叙述一致，对于$\mathbf{C}_{x_2x_2}$来说，由于
$$
\begin{aligned}
\left[ \mathbf{C}_{x_2x_2} \right]_{nm} &= E\left[ (s[n]\exp(\mathrm{j} 2 \pi f_d n) + w_2[n]) (s[m]\exp(\mathrm{j} 2 \pi f_d m) + w_2[m])^* \right] \\
&= E\left[ s[n]s^*[m] \right] \exp(\mathrm{j} 2 \pi f_d n)\exp(-\mathrm{j} 2 \pi f_d m) + E\left[ w_2[n ]w_2^*[m] \right] \\
&= r_{ss}[n - m]\exp(\mathrm{j} 2 \pi f_d (n - m)) + r_{w_2w_2}[n - m]
\end{aligned}
$$
所以
$$
\mathbf{C}_{x_2x_2} = \begin{bmatrix}
r_{ss}[0]\exp(\mathrm{j} 2 \pi f_d (0)) & r_{ss}[ - 1] \exp(\mathrm{j} 2 \pi f_d (-1)) & \cdots & r_{ss}[ - N + 1] \exp(\mathrm{j} 2 \pi f_d (- N + 1)) \\
r_{ss}[  1]\exp(\mathrm{j} 2 \pi f_d (1)) & r_{ss}[0] \exp(\mathrm{j} 2 \pi f_d (0)) & \cdots & r_{ss}[ - N] \exp(\mathrm{j} 2 \pi f_d (- N))\\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[  N - 1] \exp(\mathrm{j} 2 \pi f_d ( N - 1)) & r_{ss}[  N - 2]\exp(\mathrm{j} 2 \pi f_d ( N - 2))  & \cdots & r_{ss}[0] \exp(\mathrm{j} 2 \pi f_d ( 0))
\end{bmatrix} + \mathbf{C}_{w_2w_2}
$$
其仍然保持Toeplitz性质，说明$\mathbf{x}_2$本身还是WSS随机矢量，只不过在$\mathbf{C}_{ss}$的基础上，增加了一些具有Toeplitz性质的多普勒调制项。

对于$\mathbf{x}_1$与$\mathbf{x}_2$的互协方差矩阵来说
$$
\begin{aligned}
\left[ \mathbf{C}_{x_1x_2} \right]_{nm} &= E\left[ (s[n] + w_1[n]) (s[m]\exp(\mathrm{j} 2 \pi f_d m) + w_2[m])^* \right] \\
&= E\left[ s[n]s^*[m]\exp(-\mathrm{j} 2 \pi f_d m) \right] \\
&= r_{ss}[n - m]\exp(-\mathrm{j} 2 \pi f_d m)
\end{aligned}
$$
所以$\mathbf{C}_{x_1x_2}$有这样的形状
$$
\begin{aligned}
\mathbf{C}_{x_1x_2} &= \begin{bmatrix}
r_{ss}[0] \exp(-\mathrm{j} 2 \pi f_d 0) & r_{ss}[ - 1]\exp(-\mathrm{j} 2 \pi f_d 1) & \cdots & r_{ss}[ - N + 1] \exp(-\mathrm{j} 2 \pi f_d (N - 1)) \\
r_{ss}[  1]\exp(-\mathrm{j} 2 \pi f_d 0) & r_{ss}[0] \exp(-\mathrm{j} 2 \pi f_d 1) & \cdots & r_{ss}[ - N]\exp(-\mathrm{j} 2 \pi f_d (N - 1)) \\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[  N - 1]\exp(-\mathrm{j} 2 \pi f_d 0) & r_{ss}[  N - 2]\exp(-\mathrm{j} 2 \pi f_d 1) & \cdots & r_{ss}[0]\exp(-\mathrm{j} 2 \pi f_d (N - 1))
\end{bmatrix} \\
&= \begin{bmatrix}
r_{ss}[0] & r_{ss}[ - 1] & \cdots & r_{ss}[ - N + 1] \\
r_{ss}[  1] & r_{ss}[0] & \cdots & r_{ss}[ - N] \\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[  N - 1] & r_{ss}[  N - 2] & \cdots & r_{ss}[0]
\end{bmatrix} 
\begin{bmatrix}
\exp(-\mathrm{j} 2 \pi f_d 0) & \\
 & \exp(-\mathrm{j} 2 \pi f_d 1) \\
 && \ddots \\
 &&& \exp(-\mathrm{j} 2 \pi f_d (N - 1))
\end{bmatrix}
\end{aligned}
$$
也即对每个列进行了一个多普勒调制，等价于进行了一个列变换，这个列变换破坏了矩阵的Toeplitz性，所以$\mathbf{x}_1$与$\mathbf{x}_2$的互相关不是平稳的。

同理，如果对$\mathbf{x}_2$进行了相位的调制，也即对每个列进行相同的列变换，所以其不改变Toeplitz性质，相位调制后仍然互为平稳。

### WSS随机矢量时域/频域统计特性分析

通常来说我们将两个通道的数据拼接在一起，即$\mathbf{x} = [\mathbf{x}_1^H \mathbf{x}_2^H]$，需要分析其协方差矩阵
$$
\mathbf{C}_{xx} = \begin{bmatrix}
\mathbf{C}_{x_1x_1} & \mathbf{C}_{x_1x_2} \\
\mathbf{C}_{x_1x_2}^H & \mathbf{C}_{x_2x_2}
\end{bmatrix}
$$
通常来说，$\mathbf{C}_{xx}$的结构十分不友好，所以常常将其转换到频域，分析频域中的数据，即对时域数据做DFT
$$
X[k] = \sum\limits_{n = 0}^{N - 1} x[n] \exp(-\mathrm{j} 2 \pi \frac{k}{N} n)
$$
由此能够从$N$个时域数据得到$N$个频域数据，并且，这$N$个数据所包含的信息量是等价的，我们可以将DFT的变换表示成矩阵相乘的形式
$$
\begin{aligned}
\mathbf{X} &= \begin{bmatrix} 
X[0] \\ 
X[1] \\ 
\vdots \\
X[N - 1]
\end{bmatrix} \\
&= \begin{bmatrix}
\sum\limits_{n = 0}^{N - 1} x[n] \exp(-\mathrm{j} 2 \pi \frac{0}{N} n) \\
\vdots \\
\sum\limits_{n = 0}^{N - 1} x[n] \exp(-\mathrm{j} 2 \pi \frac{N - 1}{N} n)
\end{bmatrix} \\
&= \begin{bmatrix}
1 & 1 & \cdots & 1\\
1 & \exp(-\mathrm{j} 2 \pi \frac{1}{N} 1) & \cdots &\exp(-\mathrm{j} 2 \pi \frac{1}{N} (N - 1)) \\
\vdots & \vdots & \ddots & \vdots \\
1 & \exp(-\mathrm{j} 2 \pi \frac{N - 1}{N} 1) & \cdots & \exp(-\mathrm{j} 2 \pi \frac{N - 1}{N} (N - 1))
\end{bmatrix}
\begin{bmatrix}
x[0] \\ 
x[1] \\ 
\vdots \\
x[N - 1]
\end{bmatrix} \\
&= \mathbf{F}\mathbf{x}
\end{aligned}
$$
我们希望得到列向量$\mathbf{X}$的统计特性。首先由于$\mathbf{X}$是高斯随机变量进行线性变换而来，所以其仍然是零均值高斯随机矢量，所以现在只需要求其协方差矩阵
$$
\begin{aligned}
\mathbf{C}_{XX} &= E\left[ \mathbf{X}\mathbf{X}^H \right] \\
&= \mathbf{F}E\left[ \mathbf{x}\mathbf{x}^H \right]\mathbf{F}^H \\
&= \mathbf{F}\mathbf{C}_{xx}\mathbf{F}^H \\
\end{aligned}
$$
我们知道，若自/互协方差矩阵具有Toeplitz矩阵的结构，则其拥有近似的特征值和特征向量
$$
\lambda_i = P_{xx}\left(\frac{i}{N}\right) \\
\mathbf{v}_i = \frac{1}{\sqrt{N}} \begin{bmatrix}
1 &  \exp(\mathrm{j} 2 \pi \frac{i}{N} ) & \cdots \exp(\mathrm{j} 2 \pi \frac{i}{N}(N - 1) )
\end{bmatrix}^T
$$
即特征值就是自/互相关功率谱密度，而特征向量就是傅里叶基。

所以我们可以将DFT的变换矩阵表示为
$$
\mathbf{F} = \sqrt{N}\begin{bmatrix}
\mathbf{v}_0^H \\ 
\mathbf{v}_1^H \\ 
\vdots \\
\mathbf{v}_{N - 1}^H
\end{bmatrix}
$$
由此
$$
\begin{aligned}
\mathbf{C}_{XX} &= \mathbf{F}\mathbf{C}_{xx}\mathbf{F}^H \\
&= \mathbf{F} \sum\limits_{i = 0}^{N - 1} \lambda_i \mathbf{v}_i\mathbf{v}_i^H \mathbf{F}^H \\
&= \sum\limits_{i = 0}^{N - 1}\lambda_i \left(\mathbf{F}\mathbf{v}_i\right) \left(\mathbf{F}\mathbf{v}_i\right)^H \\
&=  N\sum\limits_{i = 0}^{N - 1}  P_{xx}\left(\frac{i}{N}\right) \begin{bmatrix}
\mathbf{v}_0^H \mathbf{v}_i \\ 
\vdots \\
\mathbf{v}_i^H \mathbf{v}_i \\ 
\vdots  \\
\mathbf{v}_{N - 1}^H \mathbf{v}_i
\end{bmatrix} \begin{bmatrix}
\mathbf{v}_i^H \mathbf{v}_0 & 
\cdots &
\mathbf{v}_i^H \mathbf{v}_i &
\cdots  &
\mathbf{v}_{i}^H \mathbf{v}_{N - 1}
\end{bmatrix} \\
&= N\begin{bmatrix}
P_{xx}\left(\frac{0}{N}\right) & & & \\
& P_{xx}\left(\frac{1}{N}\right) & & \\
& & \ddots & \\
& & & P_{xx}\left(\frac{N - 1}{N}\right)
\end{bmatrix} \\
&= N\operatorname{diag}\left[ P_{xx}\left(\frac{0}{N}\right) \cdots P_{xx}\left(\frac{N - 1}{N}\right) \right]
\end{aligned}
$$
可以看到，时域中相关的时许信息在转换为频域后，实现了去相关，协方差矩阵变为了对角阵，由此之后的计算将会更加方便。

**所以可以直接记住一个结论：无论是自相关还是互相关，如果其时域满足WSS特性，或者说时域自/互协方差矩阵满足Toeplitz性质，都可以将其转换为频域，得到一个更加简单的对角矩阵形式，从而简化之后的运算。这个结论希望以后小脑反应：时域协方差矩阵Toeplitz，频域协方差对角线。**

### TDOA频域协方差分析

同样从时域的双通道接收数据模型开始
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n - n_0] + w_2[n]
\end{aligned}
$$
转换为频域后，可以迅速得到频域的协方差矩阵
$$
\mathbf{C}_{X_1X_1} = N \operatorname{diag}\left[ P_{ss}\left(\frac{0}{N}\right) + P_{w_1w_1}\left(\frac{0}{N}\right) \cdots P_{ss}  \left(\frac{N - 1}{N}\right) + P_{w_1w_1}\left(\frac{0}{N}\right) \right]
$$
由于延迟，$\mathbf{x}_2$的相关函数保持不变，所以功率谱也保持不变，所以转换到频域后的协方差矩阵仍然与$\mathbf{C}_{X_1X_1}$相同。

也可以从另一个角度来理解时延之后的频域协方差矩阵不变这一事实，容易得到$X_2[k] = S[k] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N} n_0\right)$，所以相应的协方差矩阵为
$$
\begin{aligned}
E\left[X_2[k] X_2^*[l]\right] &=E \left[S[k] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N} n_0\right) S^*[l] \exp\left(\mathrm{j} 2 \pi \frac{l}{N} n_0\right)\right] \\
&= E\left[S[k]S^*[l]  \right] \exp\left(-\mathrm{j} 2 \pi \frac{k - l}{N} n_0\right)
\end{aligned}
$$
由于$E\left[S[k]S^*[l]  \right]$本身具只在$k = l$时有值，所以此时的相位项为0，不发生变化，最终
$$
\mathbf{C}_{X_2X_2}=\mathbf{C}_{X_1X_1}
$$
对于$\mathbf{C}_{X_1X_2}$，易证其具有Toeplitz性质，所以$\mathbf{X}_1$与$\mathbf{X}_2$互为平稳过程，拥有互相关函数和对于互功率谱密度，所以在得到互功率谱密度之后，能够迅速得到其频域互相关矩阵
$$
\begin{aligned}
E\left[ x_1(n) x_2^*(m) \right] &= E\left[ s(n) s^*(m - n_0) \right] \\
&= r_{ss}(n - m + n_0)
\end{aligned}
$$

$$
P_{x_1x_2}(f) = P_{ss}(f) \exp\left( \mathrm{j} 2 \pi f n_0 \right)
$$

所以，将互功率谱密度转换为频域协方差的对角矩阵，可以得到$\mathbf{C}_{X_1X_2}$表达式
$$
\mathbf{C}_{X_1X_2} = N \operatorname{diag}\left[ P_{ss}\left(\frac{0}{N}\right) \exp(\mathrm{j} 2 \pi \frac{0}{N} n_0) \cdots P_{ss}  \left(\frac{N - 1}{N}\right)\exp(\mathrm{j} 2 \pi \frac{N - 1}{N} n_0)  \right]
$$
同样也可以直接计算其频域互相关函数
$$
\begin{aligned}
E\left[ X_1(k) X_2^*(l) \right] &= E\left[ S(k) S^*(l)\exp\left( \mathrm{j} 2 \pi \frac{l}{N} n_0 \right) \right] \\
&= E\left[ S[k] S^*[l] \right] \exp\left( \mathrm{j} 2 \pi \frac{l}{N} n_0 \right)
\end{aligned}
$$
当且仅当$k = l$时有值，所以仍然是一个对角矩阵，保持Toeplitz性质。

**简而言之，在时域中，不管是自相关还是互相关，只要有WSS性质，那么可以直接转换为频域，写出频域的对角协方差矩阵，对角线就是功率谱密度。**

### FDOA频域协方差分析

FDOA模型为
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = s[n] \exp(\mathrm{j} 2 \pi f_d n) + w_2[n]
\end{aligned}
$$
其中，$\mathbf{x}_1$和$\mathbf{x}_2$都满足自相关WSS特性，所以可以直接得到其频域的拥有对角线性质的自相关协方差矩阵，现在重点关注如何求其频域互相关矩阵。

有前面的知识可知，FDOA模型中的时域互相关矩阵是在原本具有Toeplitz的矩阵$\mathbf{C}_{ss}$的基础上增加了列变换的多普勒调制，即每个列乘了一个多普勒相位项。这个列变换使得时域的互相关矩阵不再满足Toeplitz的结构，所以无法直接得到对角线性质的频域协方差矩阵，现在直接通过频域相关函数来得到频域互协方差矩阵
$$
\begin{aligned}
E\left[ X_1[k] X_2^*[l] \right] &= E\left[ S[k] S^*[l - d] \right] \\
\end{aligned}
$$
其中，令$\frac{d}{N} \approx f_d$，由此$s[n] \exp(\mathrm{j} 2 \pi f_d n)$的DFT为$X[k - d]$。当且仅当$k = l - d$的时候有值，所以此时的频域协方差不再是主对角线有值的矩阵，而是其他对角线有值的矩阵。
$$
\mathbf{C}_{X_1X_2} =\begin{bmatrix}
0 & \cdots & P_{ss}\left(\frac{0}{N}\right) &  & \\

&&& P_{ss}\left(\frac{1}{N}\right) & \\
&&& & \ddots & \\
&&& & & P_{ss}\left(\frac{N - 1 - d}{N}\right) \\
&&&&& \vdots \\
&&&&& 0
\end{bmatrix}
$$
从图形的记忆理解上：有了多普勒延迟之后，矩阵水平向右平移$d$个单位，出框的部分舍弃，其他为零。

更深层的内在含义为：$x_2[n]$在没有多普勒频移之前，$X_1[k]$与$X_2[k]$只有在相同频率的地方才会相关，在$x_2[n]$增加了个频移后，可以想象为其频域成分整体向右频移，自然相关关系也要向右平移，比如没频移之前，$X_1[k]$与$X_2[k]$在相同的频率$k$处相关（由此表现为对角线矩阵的形式），在频移后，$X_1[k]$要在$X_2[k + d]$处才相关，然而$x_2[n]$在整体向右平移后，仍然在$N$处截断，所以相关关系只保留了$N - d$对。

### TDOA模型频域协方差矩阵重新排列

由前面的讨论可以知道，在无论是TDOA还是FDOA模型，或是两个模型的混合，频域中无论是自协方差矩阵还是互协方差矩阵，都满足某种对角线的特性，具体来说，满足互平稳的协方差矩阵是完美的对角线矩阵，不满足互平稳的协方差矩阵也仅仅是对角线平移后的矩阵，所以可以通过重新排列以及省略部分元素的操作，使得形成一个更加方便的分块对角线形状。

若将两个通道的频域数据排列在一起，即$\mathbf{X} = \left[ \mathbf{X}_1,  \mathbf{X}_2 \right]^T$，则可以得到协方差矩阵为
$$
\mathbf{C}_{XX} = \begin{bmatrix}
\mathbf{C}_{X_1X_1} & \mathbf{C}_{X_1X_2} \\
\mathbf{C}_{X_1X_2}^H & \mathbf{C}_{X_2X_2}
\end{bmatrix} \in \mathbb{C}^{2N \times 2N}
$$
在TDOA的模型下，由于频域自相关和互相关都平稳，四个矩阵都为对角矩阵
$$
\mathbf{C}_{XX} = \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) &  &  & P_{x_1x_2}(\frac{0}{N}) &  &  \\
 & \ddots &  &  &  \ddots &  \\
 &  & P_{x_1x_1}(\frac{N - 1}{N}) &  &  & P_{x_1x_2}(\frac{N - 1}{N}) \\
P_{x_1x_2}^*(\frac{0}{N}) &  &  & P_{x_2x_2}(\frac{0}{N}) &  &  \\
 &  \ddots &  &  &  \ddots &  \\
 &  & P_{x_1x_2}^*(\frac{N - 1}{N}) &  &  & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix}
$$
单独对每个行进行分析，发现每行仅有两个元素相互相关，我们将这相互相关的元素排列到一起，即
$$
\tilde{\mathbf{X}} = \begin{bmatrix}
X_1[0] & X_2[0] & \cdots & X_1[N - 1] & X_2[N - 1] 
\end{bmatrix}^T \\
\mathbf{C}_{\tilde{X}\tilde{X}} = \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) & P_{x_1x_2}(\frac{0}{N}) &  &  &  &  \\
P_{x_1x_2}^*(\frac{0}{N}) & P_{x_2x_2}(\frac{0}{N}) &  &  &  &  \\
 &  &  & \ddots  &  &  \\
 &  &  &  &  P_{x_1x_1}(\frac{N - 1}{N}) & P_{x_1x_2}(\frac{N - 1}{N}) \\
 &  &  &  &  P^*_{x_1x_2}(\frac{N - 1}{N}) & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix} \in \mathbb{C}^{2N \times 2N}
$$
从矩阵的排列上看，在$\mathbf{C}_{XX}$中，可以同时取一个方形的四个角，然后重新压缩到一个对角块中，由此形成了$\mathbf{C}_{X'X'}$。

### FDOA模型频域协方差矩阵重新排列

FDOA的模型中，$\mathbf{C}_{X_1X_1}$和$\mathbf{C}_{X_2X_2}$仍然是对角线矩阵，但是$\mathbf{C}_{X_1X_2}$不是理想的对角线，而是向右平移的结果，排列如下
$$
\mathbf{C}_{XX} = \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) &  &  &  &  & 0 & \cdots & P'_{x_1x_2}(\frac{0}{N}) &  &  & \\
 & \ddots &  &  &  &  &  &  & \ddots &  & \\
 &  & P_{x_1x_1}(\frac{N - 1 - d}{N}) &  &  &  &  &  &  &  & P'_{x_1x_2}(\frac{N - 1 - d}{N}) \\
 &  &  & \ddots &  &  &  &  &  &  & \vdots \\
 &  &  &  & P_{x_1x_1}(\frac{N - 1}{N}) &  &  &  &  &  & 0 \\
0 &  &  &  &  & P_{x_2x_2}(\frac{0}{N}) &  &  &  &  & \\
\vdots &  &  &  &  &  & \ddots &  &  &  & \\
P'^*_{x_1x_2}(\frac{0}{N}) &  &  &  &  &  &  & P_{x_2x_2}(\frac{d}{N}) &  &  & \\
 & \ddots &  &  &  &  &  &  & \ddots &  & \\
 &  & P'^*_{x_1x_2}(\frac{N - 1 - d}{N}) & \cdots & 0 &  &  &  &  &  & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix}
$$


其中，$P_{x_1x_2}'(f)$为没有经过多普勒调制前的互平稳过程的功率谱密度，经过多普勒调制后，其频率协方差矩阵从对角线矩阵向右平移为其他对角线矩阵。

上述矩阵仍然可以重新排列，从而形成一个分块对角矩阵，仍然还是观察是否能有个方向的四个角，可以压缩到一个对角块中，观察上述矩阵，可以发现，在剔除$\mathbf{X}_1$后面$d$个元素、提出$\mathbf{X_2}$前面$d$个元素后，其仍然能够形成类似与TDOA模型这样的矩阵
$$
\begin{aligned}
\mathbf{X}_1'& = \begin{bmatrix}
X_1[0] & \cdots & X_1[N - 1 - d]
\end{bmatrix}^T \\
\mathbf{X}_2' &= \begin{bmatrix}
X_2[d] & \cdots & X_2[N - 1]
\end{bmatrix}^T
\end{aligned}
$$

$$
\mathbf{C}_{X'X'} = \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) &  &  & P'_{x_1x_2}(\frac{0}{N}) &  &  \\
 & \ddots &  &  &  \ddots &  \\
 &  & P_{x_1x_1}(\frac{N - 1 - d}{N}) &  &  & P'_{x_1x_2}(\frac{N - 1 - d}{N}) \\
P'^*_{x_1x_2}(\frac{0}{N}) &  &  & P_{x_2x_2}(\frac{d}{N}) &  &  \\
 &  \ddots &  &  &  \ddots &  \\
 &  & P'^*_{x_1x_2}(\frac{N - 1 - d}{N}) &  &  & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix} \in \mathbb{C}^{2(N - d) \times 2(N - d)}
$$

由此类似于TDOA模型的重新排列
$$
\tilde{\mathbf{X}} = \begin{bmatrix}
X_1[0] & X_2[d] & \cdots & X_1[N - 1 - d] & X_2[N - 1] 
\end{bmatrix}^T \\
\mathbf{C}_{\tilde{X}\tilde{X}} = \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) & P'_{x_1x_2}(\frac{0}{N}) &  &  &  &  \\
P'^*_{x_1x_2}(\frac{0}{N}) & P_{x_2x_2}(\frac{d}{N}) &  &  &  &  \\
 &  &  & \ddots  &  &  \\
 &  &  &  &  P_{x_1x_1}(\frac{N - 1 - d}{N}) & P'_{x_1x_2}(\frac{N - 1 - d}{N}) \\
 &  &  &  &  P'^*_{x_1x_2}(\frac{N - 1 - d}{N}) & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix} \in \mathbb{C}^{2(N - d) \times 2(N - d)}
$$
由前面所说的关于FDOA对角线形式的理解，$x_2[n]$频域向右移动了$d$个单位，所以相应的需要舍弃$\mathbf{X}_1$中后面$d$个元素以及$\mathbf{X}_2$中前面$d$个元素，才能形成$N - d$个相关对，由此形成分块对角矩阵。

### TDOA/FDOA与复衰减联合时域/频域协方差矩阵快速分析极其MLE求解

有了之前的结论，可以快速的得到在时延和多普勒同时存在的情况下，分析两个通道的相关矩阵，比如
$$
\begin{aligned}
&x_1[n] = s[n] + w_1[n] \\
&x_2[n] = a\exp(\mathrm{j}\phi)s[n - n_0]\exp(\mathrm{j} 2 \pi f_d n) + w_2[n]
\end{aligned}
$$
这个模型下，不仅增加了时延和多普勒频移，还有复衰减，一共4个未知参数
$$
\boldsymbol{\xi} = \begin{bmatrix}
a & \phi & n_0 & f_d
\end{bmatrix}^T
$$
这应该就是最接近真实情况的模型，为了对这个模型的FIM计算做准备，需要分析其时域和频域的自/互协方差矩阵，其中，第一个通道的时域频域协方差矩阵$\mathbf{C}_{x_1x_1}$和$\mathbf{C}_{X_1X_1}$无需多言。

首先分析第二个通道时域的自协方差矩阵。首先$s[n - n_0]$不会改变自相关函数；$\exp(\mathrm{j} 2 \pi f_d n)$不改变平稳特性，但是在自相关函数上增加了$\exp(\mathrm{j} 2 \pi \frac{d}{N} (n - m))$项；$a\exp(\mathrm{j} \phi)$项中的相位不会影响，幅度使得自相关函数增加$a^2$的变化。所以最终时域依旧保持平稳特性，且相关函数可以快速分析得到，自然时域协方差矩阵也能直接写出来
$$
\begin{aligned}
\left[\mathbf{C}_{x_2x_2}\right]_{nm} &= r_{x_2x_2}[n - m] \\ &= E\left[ x_2(n) x_2^*(m) \right] \\
&= a^2 r_{ss}[n - m] \exp(\mathrm{j} 2 \pi f_d(n - m))
\end{aligned}
$$
由于自相关函数仍然保持平稳特性，所以可以直接求其功率谱密度
$$
P_{x_2x_2}(f) =  a^2 P_{ss}(f - f_d) + P_{w_2w_2}(f) \\
\mathbf{C}_{X_2X_2} = \operatorname{diag}\left[P_{x_2x_2}\left( \frac{0}{N} \right), \cdots, P_{x_2x_2}\left( \frac{N - 1}{N} \right)\right]
$$
由此频域协方差矩阵也能写出来。

对于互协方差矩阵的时域和频域来说，我们首先分析$x_2[n]$中平稳的部分，也即$a \exp(\mathrm{j} \phi) s[n - n_0]$这一部分，该部分本身平稳，所以其时域自协方差矩阵可以通过求其自相关函数求得，由此也可以得到该部分频域的自协方差矩阵（自相关函数的DFT），再增加了多普勒频移项$\exp(\mathrm{j} 2 \pi f_d n)$后，其自相关函数的平稳特性不会发生改变，所以直接在原先自相关函数的基础上增加一个多普勒频移即可，由此时域和频域表达式都可求得。对于互协方差矩阵，由前面的结论，时域互协方差矩阵做了一个列变换，破坏了Toeplitz性；频域互协方差矩阵的对角线向右移动了$d$的距离。

$$
r_{ss}'[n - m] = a \exp(-\mathrm{j} \phi) r_{ss}[n - m + n_0] \\
\mathbf{C}_{x_1x_2} = a \exp(-\mathrm{j}\phi) \begin{bmatrix}
r_{ss}[n_0] & r_{ss}[n_0 - 1] & \cdots & r_{ss}[n_0 - N + 1] \\
r_{ss}[n_0 + 1] & r_{ss}[n_0] & \cdots & r_{ss}[n_0 - N] \\
\vdots & \vdots & \ddots & \vdots \\
r_{ss}[n_0 + N - 1] & r_{ss}[n_0 + N - 2] & \cdots & r_{ss}[0]
\end{bmatrix}
\begin{bmatrix}
\exp(-\mathrm{j} 2 \pi f_d 0) & \\
 & \exp(-\mathrm{j} 2 \pi f_d 1) \\
 && \ddots \\
 &&& \exp(-\mathrm{j} 2 \pi f_d (N - 1))
\end{bmatrix}
$$
频域
$$
P_{x_1x_2}'(f) = a \exp(-\mathrm{j} \phi) P_{ss}(f) \exp(\mathrm{j} 2 \pi f n_0) \\
\mathbf{C}_{X_1X_2} = a \exp(-\mathrm{j} \phi) \begin{bmatrix}
0 & \cdots & P_{ss}\left(\frac{0}{N}\right)\exp(\mathrm{j} 2 \pi \frac{0}{N} n_0) &  & \\

&&& P_{ss}\left(\frac{1}{N}\right)\exp(\mathrm{j} 2 \pi \frac{1}{N} n_0) & \\
&&& & \ddots & \\
&&& & & P_{ss}\left(\frac{N - 1 - d}{N}\right) \exp(\mathrm{j} 2 \pi \frac{N - 1 - d}{N} n_0) \\
&&&&& \vdots \\
&&&&& 0
\end{bmatrix}
$$
可以看到，时域和频域的互相关协方差矩阵相比于自相关更难分析的主要原因在于多普勒频移项$\exp(\mathrm{j} 2 \pi f_d n)$。从时域上看，其造成了时域互协方差矩阵的列变换，从而破坏了其Toeplitz性质；从频域上看，其使得频域的对角线矩阵向右平移了$d$个单位，破坏了对角线的中的非零元素，也正好对应了时域不是由WSS的相关函数傅里叶变换而来。所以综上，互相关的非平稳特性在时域上体现为被列变换破坏的Toeplitz性，频域体现矩阵中的对角线元素向右频移到了其他地方，对应了其无法由相关函数傅里叶变换而来。

现在将上述的结论直接运用在TDOA/FDOA联合估计模型的MLE求解上，由前面的知识可知

按照FDOA模型频域重新排列的方式
$$
\tilde{\mathbf{X}} = \begin{bmatrix}
X_1[0] & X_2[d] & \cdots & X_1[N - 1 - d] & X_2[N - 1] 
\end{bmatrix}^T \sim \mathcal{CN}(0, \mathbf{C}_{\tilde{X}\tilde{X}}) \\
\ln p(\tilde{\mathbf{X}}; \boldsymbol\xi) = -N \ln \pi - \ln \det\left(\mathbf{C}_{\tilde{X}\tilde{X}}\right) - \tilde{\mathbf{X}}^H \mathbf{C}_{\tilde{X}\tilde{X}}^{-1} \tilde{\mathbf{X}}
$$
其中
$$
\begin{aligned}
\mathbf{C}_{\tilde{X}\tilde{X}} &= \begin{bmatrix}
P_{x_1x_1}(\frac{0}{N}) & P'_{x_1x_2}(\frac{0}{N}) &  &  &  &  \\
P'^*_{x_1x_2}(\frac{0}{N}) & P_{x_2x_2}(\frac{d}{N}) &  &  &  &  \\
 &  &  & \ddots  &  &  \\
 &  &  &  &  P_{x_1x_1}(\frac{N - 1 - d}{N}) & P'_{x_1x_2}(\frac{N - 1 - d}{N}) \\
 &  &  &  &  P'^*_{x_1x_2}(\frac{N - 1 - d}{N}) & P_{x_2x_2}(\frac{N - 1}{N}) \\
\end{bmatrix} \in \mathbb{C}^{2(N - d) \times 2(N - d)} \\
&= \begin{bmatrix}
\mathbf{C}(f_0) & & \\
& \ddots &  \\
& & \mathbf{C}(f_{N - 1 - d})
\end{bmatrix}
\end{aligned}
$$
并且，我们已经得到
$$
\begin{aligned}
P_{x_1x_1}(f) &=  P_{ss}(f) + P_{w_1w_1}(f) \\
P_{x_2x_2}(f) &=  a^2 P_{ss}(f - f_d) + P_{w_2w_2}(f) \\
P_{x_1x_2}'(f) &= a \exp(-\mathrm{j} \phi) P_{ss}(f) \exp(\mathrm{j} 2 \pi f n_0) \\
X_1[k] &= S[k] + W_1[k] \\
X_2[k] &= a \exp(\mathrm{j} \phi) S[k - d] \exp(\mathrm{j} 2 \pi \frac{k}{N} n) + W_2[k]
\end{aligned}
$$
故
$$
\begin{aligned}
\mathbf{C}(f_k) &= \begin{bmatrix}
P_{x_1x_1}(f_k) & P'_{x_1x_2}(f_k) \\
P'^*_{x_1x_2}(f_k) & P_{x_2x_2}(f_k + f_d) \\
\end{bmatrix} \\
&= \begin{bmatrix}
P_{ss}(f_k) + P_{w_1w_1}(f_k) & a \exp(-\mathrm{j} \phi) P_{ss}(f_k) \exp(\mathrm{j} 2 \pi f_k n_0) \\
a \exp(\mathrm{j} \phi) P_{ss}(f_k) \exp(-\mathrm{j} 2 \pi f_k n_0) & a^2 P_{ss}(f_k) + P_{w_2w_2}(f_k + f_d) \\
\end{bmatrix} \\
\end{aligned}
$$

$$
\begin{aligned}
\det\left[\mathbf{C}(f_k)\right] &= P_{x_1x_1}(f_k) P_{x_2x_2}(f_k + f_d) - \left|P'_{x_1x_2}(f_k)\right|^2 \\
&= \left[ P_{ss}(f_k) + P_{w_1w_1}(f_k) \right] \left[ a^2 P_{ss}(f_k) + P_{w_2w_2}(f_k + f_d) \right] - a^2 P_{ss}^2(f_k) \\
&= P_{ss}(f_k)P_{w_2w_2}(f_k + f_d) + a^2 P_{ss}(f_k)P_{w_1w_1}(f_k) + P_{w_1w_1}(f_k)P_{w_2w_2}(f_k + f_d) \\
&\approx P_{ss}(f_k)P_{w_2w_2}(f_k) + a^2 P_{ss}(f_k)P_{w_1w_1}(f_k) + P_{w_1w_1}(f_k)P_{w_2w_2}(f_k)
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{C}^{-1}(f_k) &= \frac{1}{\det\left[\mathbf{C}(f_k)\right]}\begin{bmatrix}
P_{x_2x_2}(f_k + f_d) & -P'_{x_1x_2}(f_k) \\
-P'^*_{x_1x_2}(f_k) &  P_{x_1x_1}(f_k)\\
\end{bmatrix} \\
\end{aligned}
$$



其中，$P_{w_2w_2}(f_k) \approx P_{w_2w_2}(f_k + f_d)$，分块的子矩阵只包含参数$a$，暂时先不考虑参数$a$，则
$$
\begin{aligned}
\tilde{\mathbf{X}}^H \mathbf{C}_{\tilde{X}\tilde{X}}^{-1} \tilde{\mathbf{X}} &= \sum\limits_{k = 0}^{N - 1 - d} 
\begin{bmatrix}
X_1^*[k] & X_2^*[k + d] 
\end{bmatrix} 
\begin{bmatrix}
P_{x_2x_2}(f_k + f_d) & -P'_{x_1x_2}(f_k) \\
-P'^*_{x_1x_2}(f_k) &  P_{x_1x_1}(f_k)\\
\end{bmatrix} 
\begin{bmatrix}
X_1[k] \\ X_2[k + d] 
\end{bmatrix} \frac{1}{\det\left[\mathbf{C}(f_k)\right]} \\
&= \sum\limits_{k = 0}^{N - 1 - d} \frac{1}{\det\left[\mathbf{C}(f_k)\right]} 
\left\{ \left| X_1[k] \right|^2 P_{x_2x_2}(f_k + f_d) + \left| X_2[k] \right|^2 P_{x_1x_1}(f_k) - X_1^*[k]X_2[k + d]P'_{x_1x_2}(f_k) - X_1[k]X_2^*[k + d]P'^*_{x_1x_2}(f_k)  \right\}
\end{aligned}
$$
由于
$$
\begin{aligned}
&\left| X_1[k] \right|^2 P_{x_2x_2}(f_k + f_d) + \left| X_2[k] \right|^2 P_{x_1x_1}(f_k) \\
&= \left| X_1[k] \right|^2 \left[ a^2 P_{ss}(f_k) + P_{w_2w_2}(f_k + f_d) \right] + \left| X_2[k] \right|^2 \left[P_{ss}(f_k) + P_{w_1w_1}(f_k)\right]
\end{aligned}
$$
仅含有参数$a$，所以在不考虑参数$a$的情况下，最大似然等效于让下式最大
$$
\begin{aligned}
J &= \sum\limits_{k = 0}^{N - 1 - d} \frac{1}{\det\left[\mathbf{C}(f_k)\right]} \left\{X_1^*[k]X_2[k + d]P'_{x_1x_2}(f_k) + X_1[k]X_2^*[k + d]P'^*_{x_1x_2}(f_k)\right\} \\
&= \sum\limits_{k = 0}^{N - 1 - d} \frac{a   P_{ss}(f_k) }{\det\left[\mathbf{C}(f_k)\right]} \left\{X_1^*[k]X_2[k + d] \exp(\mathrm{j} 2 \pi f_k n_0)\exp(-\mathrm{j} \phi) + X_1[k]X_2^*[k + d] \exp(-\mathrm{j} 2 \pi f n_0)\exp(\mathrm{j} \phi) \right\} \\
&\approx N \sum\limits_{k = 0}^{N - 1 - d} a \left[g(f_k)\exp(\mathrm{j} \phi) + g^*(f_k)\exp(-\mathrm{j} \phi) \right] \frac{1}{N}\\
&\approx a T \left\{\exp(\mathrm{j} \phi)\int_{-\infty}^{\infty} g(f)  \mathrm{d} f  + \left[ \exp(-\mathrm{j} \phi)\int_{-\infty}^{\infty} g(f) \mathrm{d} f\right]^*\right\} \\
&= a T \left| \int_{-\infty}^{\infty} g(f) \mathrm{d} f \right| \cos(\alpha+\phi) \\
&= a T \left| \int_{-\infty}^{\infty} \frac{ P_{ss}(f) }{\det\left[\mathbf{C}(f)\right]} X_1(f)X_2^*(f + f_d) \exp(-\mathrm{j} 2 \pi f \tau_0) \mathrm{d} f \right| \cos(\alpha + \phi) \\
&\approx a T \left| \int_{-\infty}^{\infty} \tilde{X}_1(f)\tilde{X}_2^*(f + f_d) \exp(-\mathrm{j} 2 \pi f \tau_0) \mathrm{d} f \right| \cos(\alpha + \phi) \\
\end{aligned} \\
$$
其中
$$
g(f) = \tilde{X}_1(f)\tilde{X}_2^*(f + f_d) \exp(-\mathrm{j} 2 \pi f \tau_0) \\
\tilde{X}_1(f) = W(f)X_1(f)\\
\tilde{X}_2(f) = W(f)X_2(f) \approx W(f - f_d)X_2(f) \\
\alpha = \arg\left[\int_{-\infty}^{\infty} g(f) \mathrm{d} f\right] \\
W(f) = \sqrt{\frac{ P_{ss}(f) }{\det\left[\mathbf{C}(f)\right]}}
$$
现在不考虑参数$a$和$\phi$，则等价于下式最大
$$
\begin{aligned}
J' &= \left| \int_{-\infty}^{\infty} \tilde{X}_1(f)\tilde{X}_2^*(f + f_d) \exp(-\mathrm{j} 2 \pi f \tau_0) \mathrm{d} f \right| \\
&= \left| \int_{0}^T\tilde{x}_1(t)   \tilde{x}^*_2(t + \tau_0) \exp\left(\mathrm{j} 2\pi f_d t\right) \mathrm{d} t  \right|\\
\end{aligned}
$$
等价于信号$\tilde{x}_1(t)$与信号$\tilde{x}_2(t + \tau_0) \exp\left(-\mathrm{j} 2\pi f_d t\right)$做互相关，也即模糊函数的框架。由此得到参数$f_d$和参数$\tau_0$的最大似然估计。







