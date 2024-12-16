### 复随机变量和PDF

**复随机变量**

一个复数随机变量定义为
$$
\tilde{x} = u + \mathrm{j} v
$$
其中，$u$和$v$就是两任意的实随机变量，不同的$u$和$v$构成了不同的复随机变量$\tilde{x}$，所以应该视一个复随变量为等价的两个实随机变量。

现在定义复随机变量的统计量
$$
E[\tilde{x}] = E[u] + \mathrm{j} E[v] \\
E[|\tilde{x}|^2] = E[u^2] + E[v^2] \\
\mathrm{var}(\tilde x) = \mathrm{var}(u) + \mathrm{var}(v) = E[|\tilde{x}|^2] - |E[\tilde{x}]|^2   \\
$$
其中，二阶矩和方差从某种程度上可以视为能量，所以其方差就是代表这个复随机变量的总共能量，故为实部和虚部能量之和。

互矩定义如下（假设均值为0）
$$
\mathrm{cov}(\tilde x_1,  \tilde x_2) = E[\tilde x_1^*  \tilde x_2] = E[u_1u_2 + v_1v_2] + \mathrm{j} E[u_1v_2 - u_2 v_1]
$$
互矩的实部类似于实随机变量的相关，虚部部分可以视为两个复随机变量实部虚部相互拮抗的部分。

本质上，复随机变量就是两个实随机变量，当前没有对这两个实随机变量做任何限制，上述的统计量定义只不过也是两个实随机变量的某些统计性质。

**复高斯PDF**

为了方便运算，复高斯PDF在上述复数随机变量的基础上增加了实部和虚部必须独立同高斯分布，记作$\tilde{x} \sim \mathcal{CN}(0, \sigma^2)$，则两个独立同分布的实数高斯随机变量为
$$
u \sim \mathcal{N}(0, \sigma^2 / 2) \\
v \sim \mathcal{N}(0, \sigma^2 / 2)
$$
对于一个复随机矢量
$$
\mathbf{\tilde{x}} = \begin{bmatrix}
\tilde{x}_1 & \tilde{x}_2 & \cdots & \tilde{x}_1 
\end{bmatrix} ^T
$$
可以定义一个相关矩阵
$$
\begin{aligned}
C_{\tilde{x}}& = E[(\tilde{\mathbf{x}}-E(\tilde{\mathbf{x}}))(\tilde{\mathbf{x}}-E(\tilde{\mathbf{x}}))^H] \\
&= 

E \left\{
\begin{bmatrix}
\tilde{x}_1-E(\tilde{x}_1)\\
\tilde{x}_2-E(\tilde{x}_2)\\
\vdots\\
\tilde{x}_n-E(\tilde{x}_n)
\end{bmatrix}
\begin{bmatrix}
\tilde{x}_1^*-E^*(\tilde{x}_1) & \tilde{x}_2^*-E^*(\tilde{x}_2) & \cdots & \tilde{x}_n^*-E^*(\tilde{x}_n)
\end{bmatrix} \right\}  \\

&=\begin{bmatrix}
\operatorname{var}(\tilde{x}_1) & \operatorname{cov}(\tilde{x}_1,\tilde{x}_2) & \cdots & \operatorname{cov}(\tilde{x}_1,\tilde{x}_n)\\
\operatorname{cov}(\tilde{x}_2,\tilde{x}_1)&\operatorname{var}(\tilde{x}_2)&\ldots&\operatorname{cov}(\tilde{x}_2,\tilde{x}_n)\\
\vdots & \vdots & \ddots & \vdots\\
\operatorname{cov}(\tilde{x}_n,\tilde{x}_1)&\operatorname{cov}(\tilde{x}_n,\tilde{x}_2) & \cdots & \operatorname{var}(\tilde{x}_n)\end{bmatrix}^*
\end{aligned}
$$
表示不同复随机变量之间的相关关系。

在上述的基础上，我们可以定义一个复高斯随机矢量PDF为（假设零均值）
$$
p(\mathbf{\tilde{x}}) = \frac{1}{\pi^n \det (C_{\tilde{x}})} \exp\left( -\mathbf{\tilde{x}}^H C_{\tilde{x}}^{-1} \mathbf{\tilde{x}} \right)
$$
上述表达式虽然变量的复数，但是最后得到的结果是实数，可以证明，上述表达式等价于实随机矢量$\mathbf{x} = [u_1, \cdots u_n, v_1, \cdots v_n]^T$的PDF
$$
p(\mathbf{\tilde{x}}) = p(\mathbf{x}) = \frac{1}{(2\pi)^{n / 2} \det^{1 / 2} (C_{x})} \exp\left( -\frac{1}{2}\mathbf{x}^T C_{x}^{-1} \mathbf{x} \right) \\
C_x = \begin{bmatrix}
C_{uu} & C_{uv} \\
C_{vu} & C_{vv} \\
\end{bmatrix} = \frac{1}{2} \begin{bmatrix}
A & -B \\
B & A \\
\end{bmatrix}
$$
其中，$A = \frac{1}{2}C_{uu}$且$B = \frac{1}{2}C_{vu}$，前者代表实部或者虚部单独的序列相关，而后者代表实部和虚部的耦合信息。对于雷达信号处理来说，复信号的实部和虚部来源于独立的通道，所以实部和虚部通常是相互独立不耦合的，自然而然$C_{vu}$这项为零，所以复信号的相关矩阵依然是个实矩阵，是单独实部和虚部相关矩阵的两倍。所以从感性的认知角度来说，对于复信号的相关矩阵的认识就等价于单独实部或者单独虚部的相关矩阵。对于功率谱来说，亦是如此。

换句话说，复随机变量和实随机变量只是同一个东西不同的表现形式，使用复数的计算会更加简单，但是虽然是复数，但是本质上还是满足特定关系的实随即矢量的PDF。使用复随机变量只是为了形式上的方便。

现在证明，在$C_{uu} = C_{vv}$以及$C_{uv} = -C_{vu}$的条件下$p(\mathbf{\tilde{x}}) = p(\mathbf{x})$。该结论等效证明两个性质。P381

### filter

任何一个LTI系统的响应都可以表示为零状态响应与零输入响应之和，这里为了方便说明，将零状态响应命名为激励响应，意为系统响应单独受激励影响，而没有系统初始值的影响；零输入响应命名为初值响应，意为系统响应单独受到系统初值影响，而没有受到外来的激励影响。

根据信号与系统的内容，将系统描述为传递函数$H(z)$时，都隐含着在研究激励响应，而不关注初值响应（或者说是初值都为0），所以任何与卷积有关的操作实际上都是在计算激励响应相关的内容。

matlab中使用`filter`函数来计算一个**因果LTI系统**的响应

```matlab
y = filter(b, a, x)
```

假设滤波器的传递函数为：
$$
\begin{aligned}
H(z) &= \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}} \\
&= \frac{\sum\limits_{k = 0}^{M} b_k z^{-k}}{\sum\limits_{k = 0}^{N} a_k z^{-k}}
\end{aligned}
$$
等价地，该LTI系统也可以用差分方程描述
$$
\sum\limits_{k = 0}^{N} a_k y(n - k) = \sum\limits_{k = 0}^{M} b_k x(n - k)
$$
`filter`使用差分方程递推来得到系统的输入的，默认情况下不给初值就计算该系统的激励响应。

`filter`输出一个与输入相同长度的向量，若滤波器是FIR，且长度为$M$；输入信号长度为$N$，那么`filter`最后得到两个序列线性卷积后的结果（长度$N + M - 1$）的前$N$个元素；

若滤波器是IIR，则其冲激响应无限长，由于前面做了因果LTI的限定，所以其是在正半轴的无限长。若输入的长度为$N$，理论上的激励响应是无穷长的，但是filter会截取前$N$个元素并返回。

所以若是想到得到滤波器`b`和`a`所描述的滤波器的冲激响应，可以输入长度为$N$的冲激信号（第一个元素为1，后面的$N - 1$元素为0），由此便可以得到冲激函数的前$N$个值，若$N$超过了FIR滤波器系数的长度，则超过的部分补零；若没超过，则对FIR的系数的前$N$个元素进行了截取。因为IIR的冲激响应无限长，所以最终一定是对IIR冲激响应的截取。

### periodogram

如果$N = 512$，如果按照正常的使用FFT去做周期图，则需要将$\lfloor N / 2 \rfloor = 512$的点归为负频率，因为`fftshift`函数就是这样做的，但是`peridogram`函数将$\lfloor N / 2 \rfloor + 1 = 513$个点归为正频率，也即他认为$f = 256 / 512$这个频率属于正频率，也没错，只不过平时使用`fftshift`将这个点归为负频率。

### 频域补零（非整数延迟）

DFT的正变换和反变换公式为
$$
\begin{aligned}
X[k] &= \sum\limits_{n = 0}^{N - 1}x[n] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N}n\right) \\
x[n] &= \sum\limits_{k = 0}^{N - 1}X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} n\right) \\
\end{aligned}
$$
实际上为了显示方便，应该是时域范围的$0 \leq n \leq N - 1$与频域范围的$-N / 2 \leq k \leq N / 2 - 1$对应（假设$N$为偶数）。然而为了计算方便，时域范围的$0 \leq n \leq N - 1$与频域范围的$0 \leq k \leq N - 1$相对应，但是其仍然可以通过傅里叶变换对周期序列的特性还原出方便表示的频域范围，也就是`fftshift`函数。

实际上，可以稍微变换一下IDFT的公式
$$
\begin{aligned}
x[n] &= \sum\limits_{k = 0}^{N - 1}X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} n\right) \\
&= \sum\limits_{k = 0}^{N - 1}X[k] \exp\left(\mathrm{j} 2 \pi \left(\frac{k}{N'}\right) \left(\frac{N'}{N}n\right)\right)
\end{aligned}
$$
同时，我们在$X[k]$的末尾（低频处）补零至频域的范围为$0 \leq k \leq N' - 1$的序列$X'[k]$，并对其做IDFT
$$
\begin{aligned}
x'[n'] &= \sum\limits_{k = 0}^{N' - 1}X'[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N'} n'\right) \\
&= \sum\limits_{k = 0}^{N - 1}X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N'} n'\right) \\
\end{aligned}
$$
其中，$0 \leq n' \leq N' - 1$。

对比上面两个式子，可以得到
$$
x[n] = x'[\frac{N'}{N}n]
$$
其中，$x[n]$是原时域序列，而$x'[n]$是$X[k]$末尾补零的序列，可以视为对$x[n]$进行了某种方式的插值，拓展到了$[0 : N' - 1]$的范围。所以这个式子的含义就是：**原时域序列可以视为频域补零后的时域插值序列的等间隔抽取**。

那么现在关系的问题是频域末尾补零究竟对应于时域进行怎样的插值。

- 两端补零相当于低频补零，也就是原来信号所有的频率分量都等比例增加，比如原来的DC电平信号变成了一个正弦信号；
- 末尾补零可以视为两端补零的循环移位，而循环移位的物理含义在于所有频率以同样的频率间隔平移，所以末尾补零首先进行了一个等比例频率增加，再进行一个等间隔平移；
- 这是一个复杂的变换，可以视为对原序列进行了一个复杂且具有提高频率性质的插值。

这个插值后图形的效果没有很大的作用，但是由于matlab中`ifft`函数极大概率会自动末尾补零，所以是一个常见的错误。

有了这些知识，可以来讨论一下非整数频移的效果。由DFT的循环移位性质可知
$$
\begin{aligned}
x[n - n_0]  &\leftrightarrow  X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} n_0\right) \\
? &\leftrightarrow X[k] \exp\left(\mathrm{j} 2 \pi \left(\frac{k}{N}\right) \tau_0\right)
\end{aligned}
$$
同样可以理解为，在频域上进行一个多普勒调制，当$n_0$是整数，可以很轻易得到时域就是循环移位的结果，如果是不为整数的$\tau_0$，其对应的时域是怎样的呢？
$$
\begin{aligned}
X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} \tau_0\right)
&= X[k] \exp\left(\mathrm{j} 2 \pi \left(\frac{k}{N}\right) \left(  \frac{N}{N'}n_0\right)\right)\\
&= X'[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N'} n_0\right)\\
&\leftrightarrow x'[n - n_0]
\end{aligned}
$$
首先，令$\tau_0 = \frac{N}{N'}n_0$，$n_0$为整数，所以可以将$\tau_0$用整数表达了出来。

其次，我们将$X[k]$进行末尾补零的操作。可以发现，由于$n_0$是整数，末尾补零后再转到时域，相当于时域经过插值后的整数循环移位。

又由于原始序列就是插值序列的等间隔抽取，所以非整数的循环移位等价进行：**时域插值，循环移位，再等间隔抽取**。然而，该方式直接向$X[k]$末尾补零，导致的时域插值十分奇怪，所以是一个常见的错误。

我们正真的目的是在频域进行非整数的多普勒调制，使得时域能够循环移位至正确的插值结果。前面所述的插值方式是低频补零，这是不正确的，我们对频域序列进行高频补零
$$
\tilde{X}[k] = \begin{cases}
X[k], \quad 0\leq k \leq \frac{N}{2} - 1\\
0, \quad \frac{N}{2} \leq k \leq  N' - \frac{N}{2} - 1 \\
X[k - N' + N], \quad N' - \frac{N}{2} \leq k \leq N' - 1
\end{cases}
$$
其中，$N'$是补零后的总点数。然后再补零频域序列的基础上转为时域插值序列
$$
\tilde{X}[k] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N'}n_0\right) = \tilde{X}[k] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N}\tau_0\right)\leftrightarrow \tilde{x}[n - n_0]
$$
最终在再对$\tilde{x}[n - n_0]$进行等间隔抽样即可，**流程复杂，但是好理解**。

若时域序列$x[n]$的范围为$[-N/2, N/2-1]$，对应的频域范围为$[-N/2, N/2-1]$，此时的DFT正反变换为
$$
\begin{aligned}
X[k] &= \sum\limits_{n = -N / 2}^{N / 2 - 1}x[n] \exp\left(-\mathrm{j} 2 \pi \frac{k}{N}n\right)  \\
x[n] &= \sum\limits_{k = -N / 2}^{N / 2 - 1}X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} n\right)
\end{aligned}
$$
其中，$-N / 2 \leq k \leq N / 2 - 1$，两边高频，中间低频。

同样在$X[k]$的末尾补零得到$X'[k]$，再转回时域，此时时频两域的范围都是$[-N / 2, N' - N / 2 - 1]$。
$$
\begin{aligned}
x'[n] &= \sum\limits_{k = -N / 2}^{N' - N / 2 - 1}X'[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N'} n\right) \\
&= \sum\limits_{k = -N / 2}^{N / 2 - 1}X[k] \exp\left(\mathrm{j} 2 \pi \frac{k}{N} \left( \frac{N}{N'} n \right) \right) \\\end{aligned}
$$
也即
$$
x[n] = x'[\frac{N}{N'}n]
$$
原时域序列仍然是频域补零后的时域插值序列的抽取，然而此时的频域补零对应的是高频补零，所以时域的插值就是原时域同等频率的细化，是正真有用的插值。如果进行了非整数的多普勒频移，对应着插值序列的循环移位，再抽取。

也就是说，要实现同频率的插值的核心在于频域的高频插值，也就是要化成频域范围为$[-N/2, N/2-1]$的时候在进行非整数多普勒频移，然后再返回时域。

为了在matlab上实现高频补零的时域插值，函数`fft`默认都是在范围$[0, N-1]$进行的，所以需要进行一些等价的操作，步骤如下

- 得到一个$[0 : N-1]$的序列；
- 对于使用`fft`函数，得到$[0 : N-1]$的频域序列；
- 对频域序列做`fftshift`转移到频域范围$[-N / 2 : N / 2 - 1]$；
- 在频域范围$[-N / 2 : N / 2 - 1]$处进行非整数的多普勒频移（小数导致了自动进行了高频补零）；
- 使用`ifftshift`将频域范围转至$[0 : N-1]$；
- 使用`ifft`函数。

代码如下
```matlab
N = length(x_n);
n = 0 : N - 1;
n_ = n - floor(N / 2);

X_k = fftshift( fft(x_n) );
X_k_tau = X_k .* exp(-1j * 2 * pi * n_' * tau / N);
X_k_tau = ifftshift(X_k_tau);
x_n_tau = ifft(X_k_tau);

phi = 0;
x_n_tau_dopler = exp(1j * phi) * (x_n_tau .* exp(1j * w_d * n_'));
```

最后进行多普勒频移，要清楚你针对的时间轴是在$[0 : N - 1]$还是$[-N / 2 : N / 2 - 1]$。

该程序操作等效于论文中的$\mathbf{s}_2 = \mathbf{Q} \mathbf{s}_1$。