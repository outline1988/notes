参考：Sparse Methods for Direction-of-Arrival Estimation

张颢 现代数字信号处理2 （前两讲中关于零范数不确定性的部分没记、第四讲后半部分贪婪算法的剩余部分没看、第三讲后半部分非平滑优化的内容没看）

### Sparse Representation

考虑一个线性模型
$$
\mathbf{y} = \mathbf{A} \mathbf{x}
$$
其中，$\mathbf{A} \in \mathbb{R}^{M \times \bar{N}}$是一个宽胖矩阵（dictionary），即$M \ll \bar{N}$。所以多数情况下，有无数满足该线性模型的解（不考虑少数$\mathbf{y}$没有落在$\mathbf{A}$列空间的情况）。稀疏表征（sparse representation）即使用少数$\mathbf{A}$中的列（称为atom）来线性表示$\mathbf{y}$，也即$\mathbf{x}$是稀疏的。

稀疏恢复（sparse recovery）在给定观测$\mathbf{y}$和字典$\mathbf{A}$的情况下，恢复出最为稀疏的$\mathbf{x}$。在数学上等价求解一个优化问题
$$
\min_{\mathbf{x}} \|\mathbf{x} \|_0, \; \text{s.t.} \; \mathbf{y} = \mathbf{A}\mathbf{x},
$$
其中，零范数$\|\mathbf{x} \|_0$表示$\mathbf{x}$中非零元的个数，即
$$
\|\mathbf{x} \|_0 = \# \{ i: x_i \neq 0 \}. 
$$
定义$\mathrm{spark}(\mathbf{A})$，即任意$\mathrm{spark}(\mathbf{A}) - 1$数量抽取$\mathbf{A}$中列向量，都是线性无关的。这个时候，如果你找到了一个$\mathbf{x}$满足
$$
\|\mathbf{x} \|_0 = K < \frac{\mathrm{spark}(\mathbf{A})}{2},
$$
那么$\mathbf{x}$就是就是该优化问题的解， 不存在零范数小于或等于$K$且还满足线性方程组的解$\mathbf{x}'$了。若存在，则必有$\mathbf{A}(\mathbf{x} - \mathbf{x}') = \mathbf{0}$，而$\| \mathbf{x} - \mathbf{x}' \|_0 \leq \| \mathbf{x} \|_0 + \| \mathbf{x}' \|_0 \le 2K < \mathrm{spark}(\mathbf{A})$，即$\mathbf{A}$的零空间中存在$\mathrm{spark}(\mathbf{A}) - 1$（或更少）非零元的向量$\mathbf{x} - \mathbf{x}'$，与$\mathrm{spark}(\mathbf{A})$的定义相矛盾。

这个问题是NP-hard问题，直接求解很难。

### Convex Relaxation

求解稀疏恢复问题的一个途径是进行凸松弛（convex relaxation），将$\ell_0$松弛到$\ell_1$，后者是凸的。即求解另一个优化问题
$$
\min_{\mathbf{x}} \|\mathbf{x} \|_1, \; \text{s.t.} \; \mathbf{y} = \mathbf{A}\mathbf{x},
$$
也称为BP（basis pursuit）问题。

并不是所有的稀疏恢复问题能够将$\ell_0$松弛到$\ell_1$，同时保持两个问题具有相同的解。需要对$\mathbf{A}$以及真值$\mathbf{x}$有一定的要求，不正式的说，字典$\mathbf{A}$越正交，真值$\mathbf{x}$越稀疏，$\ell_0$越能转换为$\ell_1$。

描述$\mathbf{A}$的正交性的第一个工具是mutual coherence，即定义一个关于$\mathbf{A}$的量$\mu(\mathbf{A})$
$$
\mu(\mathbf{A}) = \max_{i \neq j} \frac{\left| \langle \mathbf{a}_i, \mathbf{a}_j \rangle  \right|}{ \| \mathbf{a}_i \|_2 \| \mathbf{a}_j \|_2}.
$$
$\mu(\mathbf{A})$越小，表示矩阵$\mathbf{A}$越正交，理解为矩阵内相关程度最高的两个向量的相关。

正式描述为：若$\mu(\mathbf{A}) < \frac{1}{2K - 1}$，则能将真值零范数$\| \mathbf{x} \|_0 \le K$的$\ell_0$优化问题转为BP问题。

描述$\mathbf{A}$的正交性的另一个工具是RIP（Restricted Isometry Property）。定义一个关于$\mathbf{A}$的K-restricted isometry constant（RIC）$\delta_K$，对于任意$K$-sparse向量$\mathbf{v}$，都满足
$$
(1 - \delta_K) \| \mathbf{v} \|_2 \le \| \mathbf{A} \mathbf{v} \|_2  \le (1 + \delta_K)\| \mathbf{v} \|_2
$$
$\delta_K$越小，表示任意取$K$个$\mathbf{A}$中的向量，正交程度越高。

正式描述为：若$\delta_{2K} < \sqrt{2} - 1$，即任意$2K$的$\mathbf{A}$中向量，都满足一定程度的正交性，则能将真值零范数$\| \mathbf{x} \|_0 \le K$的$\ell_0$优化问题转为BP问题。

所以，在凸松弛后，新的优化问题变为了

- BP问题（这里再写一遍）
  $$
  \min_{\mathbf{x}} \|\mathbf{x} \|_1, \; \text{s.t.} \; \mathbf{y} = \mathbf{A}\mathbf{x}.
  $$

- 实际中，观测$\mathbf{y}$存在噪声，即$\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{e}$，所以求解中需要考虑到噪声，变为了BPDN问题
  $$
  \min_{\mathbf{x}} \|\mathbf{x} \|_1, \; \text{s.t.} \; \| \mathbf{y} - \mathbf{A}\mathbf{x} \|_2 \leq \eta.
  $$
  其中，$\eta \ge \| \mathbf{e} \|_2$是与噪声方差有关的一个上界。如果$\mathbf{e}$是$N$维的高斯白噪声，则$\| \mathbf{e} \|_2 = \sqrt{N} \hat{\sigma}$。

- 再将BPDN转换为无约束问题（拉格朗日），变为了LASSO（Least Absolute Shrinkage and Selection Operation）问题
  $$
  \min_\mathbf{x} \left\{ \frac{1}{2} \| \mathbf{y} -  \mathbf{A}\mathbf{x} \|_2^2 + \lambda \|\mathbf{x} \|_1 \right\}. 
  $$

在$\eta$和$\lambda$的选择下，BPDN和LASSO问题等价，都是在解决一定噪声容限的情况下的稀疏恢复问题。当$\eta = 0$且$\lambda \to 0^{+}$的情况下，BPDN和LASSO都等价于BP问题。

在实际中，为了方便求解，具体都是求解LASSO问题，同时是知道噪声方差的。所以如何通过噪声方差得到一个合适的$\lambda$，进而求解LASSO也是一个问题。再选择合适的$\lambda$后，就是一个无约束优化问题，目标函数的non-smooth的，所以需要使用相关的方法来进行求解（主要是次梯度下降，这个内容超出了本笔记的范围）。

### Discrepancy Principle

参考：A sparse signal reconstruction perspective for source localization with sensor arrays 5.4

对于BPDN问题，其中的约束参数$\eta$是一个与噪声$\mathbf{e}$方差有关的一个上界，那么究竟该如何选择这个具体的值呢？

若$\eta$特别小，意味着我们对于数据的拟合十分苛刻，只允许有很小的误差。所以一个更精确（更多非零元）的$\mathbf{x}$就会恢复出来，最终在谱中呈现为宽主瓣和多峰（因为更多的字典参与进来完成对噪声的拟合），这是我们不想要的；若$\eta$特别大，也即我们对于数据的拟合十分宽松，这样就更加促进了恢复信号的稀疏度，也即用到了更少的字典来实现对于数据的拟合，这也是我们不想要的。

因为$\| \mathbf{e} \|_2$本身是一个随机变量，不同的观测数据在数值上都不同，服从某一特定分布。最好的$\eta$选择就是紧贴着当前数据所实现的$\| \mathbf{e} \|_2$，然而这一数值无法得到，所以我们退而求其次，牺牲一点点的紧贴度（放松一点拟合程度），找到一个$\eta$，使其有0.999的概率会超过$\| \mathbf{e} \|_2$的某次实现，即
$$
\eta \ge \| \mathbf{e} \|_2
$$
我们仅仅只是放松了一点点的数据拟合程度，在实际中，毫不影响。由此，我们需要掌握关于$\| \mathbf{e} \|_2$的分布。

我们知道，如果$\mathbf{e}$中的每一个元素都是一个独立同分布方差为$\sigma^2$的复高斯随机变量，那么$M$个复随机变量的平方和相当于$2M$个方差一半的实随机变量平方和，最终得到一个$\chi_{2M}^2$分布
$$
\frac{1}{\sigma^2 / 2} \sum\limits_{i = 1}^{M} |e_i|^2 = \frac{1}{\sigma^2 / 2} \sum\limits_{i = 1}^{M} (\bar{e}_i^2 + \tilde{e}_i^2) \sim \chi^2_{2M}
$$
所以最后的选择就是在这个分布的CDF在0.999的位置卡一个值，具体使用matlab的函数`chi2inv`或者`gaminv`。注意到，缩放的chi方分布就是Gamma分布
$$
\frac{\sigma^2} {2} \chi_{2M}^2 = \text{Gamma}(\frac{2 M}{2}, 2 \frac{\sigma^2}{2})
$$

在约束门限的确定中，前面的讨论只考虑噪声的因素，然而，即使是无噪声的观测数据，在使用稀疏恢复时也会包含着噪声（可能是数值精度或计算方法等各种各样方式引起的误差），暂且叫做恢复噪声（其实是零阶近似误差，或者网格误差）。恢复噪声与信号的功率有关，信号功率越大，恢复噪声就会越大。所以在稀疏恢复的时候，如果固定噪声功率，当SNR很高时，单单由噪声分布得到的门限是不够的（在低SNR时，恢复误差相比于噪声误差微不足道），所以会有多峰的现象。

以后再来研究这个恢复误差吧。

关于$l_1$-SVD的门限的确定也需要讨论。


### Greedy

解决稀疏恢复问题的另一个途径是贪婪算法。稀疏度从大到小，不断地挑选最有可能的一组向量，步骤如下

- 对于$\mathbf{A} \in \mathbf{R}^{M \times \bar{N}}$中的任意向量$\mathbf{a}_i$，计算$z_i = | \mathbf{a}_i^T \mathbf{y} |$；
- 取最大的$K$个$z_i$所对应的$K$个$\mathbf{a}_i$，使用最小二乘即可恢复出系数；
- 这个$K$可以是预先确定的，也可以通过比较$\mathbf{y}$与$\mathbf{a}_i$的距离小于与噪声方差相关的某一门限来确定。

假设某一稀疏恢复问题的真值零范数就是$\| \mathbf{x} \|_0 = K$，那么该方法能work的前提在于这真值所对应的原子与$\mathbf{y}$相关后的值恰好就是最大的$K$个值。假设原子$\mathbf{a}_i$已经进行了归一化，$\|\mathbf{a}_i \|_2 = 1$，假设集合$\mathcal{S} = \{ i : x_i \neq 0 \}$，那么数学上就是要满足以下条件
$$
\min_{i \in \mathcal{S}} z_i \ge \max_{i \notin \mathcal{S}} z_i
$$
若$i \in \mathcal{S}$，则有
$$
\begin{aligned}
z_i &= \left|\mathbf{a}_i^T \mathbf{y}\right| = \left|\mathbf{a}_i^T \sum\limits_{j \in \mathcal{S}} \mathbf{a}_j x_j \right| = \left|x_i + \mathbf{a}_i^T \sum\limits_{\substack{j \in \mathcal{S} \\ j \neq i}} \mathbf{a}_j x_j \right| \\
&\le \left| x_i \right| -  \left| \sum\limits_{\substack{j \in \mathcal{S} \\ j \neq i}}\mathbf{a}_i^T \mathbf{a}_j x_j \right| \le \left| x_i \right| -  \sum\limits_{\substack{j \in \mathcal{S} \\ j \neq i}} \left| \mathbf{a}_i^T \mathbf{a}_j  \right|  \left| x_j \right| \\
&\le \left| x_i \right| -  \mu \sum\limits_{\substack{j \in \mathcal{S} \\ j \neq i}} \left| x_j \right| = \left| x_i \right| -  \mu \left( \| \mathbf{x} \|_1 - x_i \right)
\end{aligned}
$$
若$i \notin \mathcal{S}$，则有
$$
\begin{aligned}
z_i &= \left|\mathbf{a}_i^T \mathbf{y}\right| = \left|\mathbf{a}_i^T \sum\limits_{j \in \mathcal{S}} \mathbf{a}_j x_j \right| \le \sum\limits_{j \in \mathcal{S}} \left| \mathbf{a}_i^T \mathbf{a}_j  \right|  \left| x_j \right| \\
& \le \mu \| \mathbf{x} \|_1
\end{aligned}
$$
由此，需要满足
$$
\left| x_i \right| -  \mu \left( \| \mathbf{x} \|_1 - x_i \right) \ge \mu \| \mathbf{x} \|_1 \\
\frac{\left| x_i \right|}{\| \mathbf{x} \|_1} \ge \frac{2 \mu}{\mu + 1}
$$
用每一个$\mathbf{a}_i$与$\mathbf{y}$做相关，我们希望在$\mathcal{S}$内的原子保留有最大的相关，这是显然的，因为$\mathbf{y}$就是由$\mathcal{S}$内的原子线性组合而成。同时，希望在$\mathcal{S}$外的原子保由尽量小的相关，显然需要对$\mathbf{A}$的正交性有所要求，所以$\mu$越小越好。其次，我们希望真值内非零元素的占比尽量均匀，因为如果差异过大，有可能在$\mathcal{S}$之外的原子与占比大的非零元素对应的原子相关（在占比大的$x_i$作用下）超过非零元素占比小的原子自身的相关（在占比小的$x_i$作用下），所以同样需要对真值$\mathbf{x}$各个元素内的占比做出要求，要求越平均越好。显然，第二个条件在实际中是不合理的。

### OMP

为此，在上述方法的基础上，提出OMP（Orthogonal Matching Persuit）方法。

使用$z_i$来得到原子的方法，首先有一个观察，在有一定正交性的情况，最大的$z_i$所对应的原子一定是符合的。所以，每次我们只寻找一个最大相关所对应的原子。再找下一个原子时，使用正交投影的方式将之前找到过的原子去掉，直到残差满足噪声方差的要求。具体的流程为

- $\mathcal{T}_0 = \emptyset$；
- step 1: $\mathbf{r}_1 = \mathbf{P}_{\mathcal{T}_0}^{\perp} \mathbf{y}$，$k_1 = \arg \max\limits_{i} |\mathbf{a}_i ^T \mathbf{r}_1|$, $\mathcal{T}_1 = \mathcal{T}_0 \cup \{ \mathbf{a}_{k_1} \}$; 
- step 2: $\mathbf{r}_2 = \mathbf{P}_{\mathcal{T}_1}^{\perp} \mathbf{y}$，$k_2 = \arg \max\limits_{i} |\mathbf{a}_i ^T \mathbf{r}_2|$, $\mathcal{T}_2 = \mathcal{T}_1 \cup \{ \mathbf{a}_{k_2} \}$; 
- step $j$: $\mathbf{r}_j = \mathbf{P}_{\mathcal{T}_{j - 1}}^{\perp} \mathbf{y}$，$k_j = \arg \max\limits_{i} |\mathbf{a}_i ^T \mathbf{r}_j|$, $\mathcal{T}_2 = \mathcal{T}_1 \cup \{ \mathbf{a}_{k_2} \}$; 
- until $\| \mathbf{y} -\mathbf{P}_{\mathcal{T}_{j}} \mathbf{y} \| \leq \eta$, where $\eta$ is a threshold determined by the noise variance.

注意到，投影将$\mathbf{y}$中的特定向量进行了移除（虽然也会对其他向量造成影响，但是在这里这种影响没有关系），即
$$
\begin{aligned}
\mathbf{r}_j &=  \mathbf{P}_{\mathcal{T}_{j - 1}}^{\perp} \mathbf{y} \\ 
&= \mathbf{P}_{\mathcal{T}_{j - 1}}^{\perp} \sum\limits_{j \in \mathcal{S}} \mathbf{a}_j x_j \\
&= \sum\limits_{j \in \mathcal{S}} \mathbf{P}_{\mathcal{T}_{j - 1}}^{\perp}\mathbf{a}_j x_j \\
&= \sum\limits_{\substack{j \in \mathcal{S} \\ j \neq k_1, \cdots k_{j - 1}}} \mathbf{a}'_j x_j
\end{aligned}
$$
其中，$\mathbf{a}'_j = \mathbf{P}_{\mathcal{T}_{j - 1}}^{\perp}\mathbf{a}_j$，即便投影之后，$\mathbf{a}'_j$也仍然保留有相关性。

OMP只要求字典$\mathbf{A}$具有一定的正交程度。


张颢的稀疏信号处理第三第四讲后半部分关于近邻优化的内容都没有看。

















