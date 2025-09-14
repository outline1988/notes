### FMCW与脉冲体制LFM处理机制对比

从脉冲体制雷达的脉压处理开始，假设接收的信号为$s[n]$只在$[n_0, N + n_0 - 1]$的范围有值，参考信号$r[n]$采样自连续的LFM信号$r(t)$
$$
r(t) = \exp(\mathrm{j} \pi \mu t^2), \;\; t \in [0, T], \;\; \mu = \frac{B}{T}
$$

$$
\begin{aligned}
r[n] &= r(nT_s) = \exp(\mathrm{j} \pi \mu T_s^2 n^2) \\
&= \exp(\mathrm{j} \pi \frac{1}{N} n^2), \;\; 0 \le n \le N - 1
\end{aligned}
$$

其中，$T$是脉宽；$B$是带宽；$T_s = 1 / B$为采样率；$\mu T_s^2 = 1 / (BT) = 1 / N$，这也说明了时宽带宽积的意义就是采样点数。

对$s[n]$使用参考信号$r[n]$进行滑窗相关（匹配滤波）
$$
\begin{aligned}
y[k] &= \sum\limits_{n = -\infty}^{\infty} s[n] r^*[n - k] \\
\end{aligned}
$$
其中
$$
\begin{aligned}
r^*[n - k] &= \exp(-\mathrm{j} \pi \frac{1}{N} (n - k)^2) \\
&= \exp(-\mathrm{j} \pi \frac{1}{N} n^2) \exp(\mathrm{j} 2 \pi \frac{1}{N} nk) \exp(-\mathrm{j} \pi \frac{1}{N} k^2)
\end{aligned}
$$
由此
$$
\begin{aligned}
y[k] &= \sum\limits_{n \in \mathcal{S}} s[n] \exp(-\mathrm{j} \pi \frac{1}{N} n^2) \exp(\mathrm{j} 2 \pi \frac{1}{N} nk) \exp(-\mathrm{j} \pi \frac{1}{N} k^2) \\
&= \exp(-\mathrm{j} \pi \frac{1}{N} k^2) \sum\limits_{n \in \mathcal{S}} \tilde{s}[n]  \exp(\mathrm{j} 2 \pi \frac{k}{N} n)
\end{aligned}
$$
其中，求和的范围为使得$s[n]$和$r[n]$都有值的部分$\mathcal{S}$。$\tilde{s}[n] = s[n] \exp(-\mathrm{j} \pi \mu T_s^2 n^2)$，范围为$\mathcal{S}$。可以将其解释为对$s[n]$在范围$\mathcal{S}$中做了一个数字下混频（DDC），下混频的参考信号就是$r[n]$，得到的信号$s'[n]$再对其作采样点数为$N$的离散傅里叶变换，范围$\mathcal{S}$不够$N$的部分补零。

简单来说，$y[k]$一种意义下是$r[n]$右移滑窗$k$点时与$s[n]$的相关。在LFM波形的情况下，可以解释为在$\mathcal{S}$（要么$s[n]$的前半段，要么后半段），对$\tilde{s}[n]$做特定频率$k$的$N$点DFT。随着$k$的变化，$\tilde{s}[n]$的波形不会变化，但是对$\tilde{s}[n]$进行DFT的范围会发生变化，同时DFT的频率$k$也发生变化。随着$k$越靠近$s[n]$的范围，$\tilde{s}[n]$做DFT的长度会越来越完全，同时频率也越来越对准$\tilde{s}[n]$的频率。

FMCW的处理思路借鉴脉冲LFM的滑窗匹配滤波的第二种理解，本质上是对滑窗匹配滤波的近似。FMCW处理先对接收信号做下混频（模拟域），再对单频信号采样。近似在于，波形没有匹配时，前面的$\tilde{s}[n]$做DFT只能在随着$k$不同而不同的的$\mathcal{S}$做DFT，而FMCW的处理一直在同一个范围（$r[n]$与$s[n]$的重合部分）中，这点区别没有什么好与不好的分别。其二在于，波形在完全匹配时，由于$r[n]$与$s[n]$的重合部分有一段缺失，所以有部分的波形失配，但是这引起的信噪比的损失认为可以忽略。

FMCW处理的方式还有一个优点，即对单频信号的采样率的要求不一定为带宽，而是可以更低，也能够获得与滑窗相关脉冲压缩获得相同的距离分辨率，比带宽更低的采样率的代价是最大可检测的单频信号的频率受到限制，但这是可以接受的，一是实际中往往知道目标的距离不会太远（或者说$T$很大）；二是如果太远了，使用下混频测单频的方式误差也很高。

从滑窗相关中DFT的部分入手
$$
\sum\limits_{n \in \mathcal{S}} \tilde{s}[n]  \exp(\mathrm{j} 2 \pi \frac{k}{N} n)
$$
其中，$\mathcal{S}$的范围为$N$左右。因为$\tilde{s}[n]$是差频信号，所以最高频率相比于最开始的$B$要小很多，所以我们能够使用interpolation和decimation的方式进行分数因子的采样率转换，由此形成一个新的序列$\tilde{s}'[n]$，长度大概为$N' < N$。此时的频谱
$$
\sum\limits_{n \in \mathcal{S}} \tilde{s}'[n]  \exp(\mathrm{j} 2 \pi \frac{k}{N'} n)
$$
在$k = 0, \cdots ,N' - 1$的频谱是与没下采样之前是相同的，所以只要下采样后采样率仍然满足差频信号的奈奎斯特定理，是可行的。只不过原先滑窗搜索搜索到$N$个点，现在只搜索到$N'$个点。FMCW就是对差频采用一个更低采样率，近似等价于滑窗前$N'$个点。



### 双基地雷达和单基地雷达几何上的区别和联系

