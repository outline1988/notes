### On-Grid Sparse DOA Methods

参考：Sparse Methods for Direction-of-Arrival Estimation

A sparse signal reconstruction perspective for source localization with sensor arrays（这个还没认真看）

若DOA估计的模型为
$$
\mathbf{y}(t) = \mathbf{A}(\boldsymbol{\theta}) \mathbf{s}(t) + \mathbf{e}(t), \\
\boldsymbol{\theta} = \begin{bmatrix}
\theta_1 &  \cdots & \theta_K
\end{bmatrix}
$$
为了使用稀疏恢复的方法，一个很自然的想法就是将搜索空间进行网格化
$$
\mathbf{y}(t) = \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{x}(t) + \mathbf{e}(t), \\
\bar{\boldsymbol{\theta}} = \begin{bmatrix}
\theta_1 &  \cdots & \theta_{\bar{N}}
\end{bmatrix}
$$
其中，$\bar{N} \gg K$，由此$\mathbf{A}(\bar{\boldsymbol{\theta}})$是一个宽胖的字典，而由于$\mathbf{y}(t)$是$K$个对应DOA的导向矢量的线性组合，所以$\mathbf{x}(t)$具有了稀疏性
$$
x_{\bar{n}}(t) = \begin{cases}
s_k(t), & \theta_{\bar{n}} = \theta_k \\
0, & \text{others}
\end{cases}
$$
需要注意的是，上述结果只有当$\theta_{\bar{n}} = \theta_k$成立时才能严格成立，即真实的DOA需要完美落到预设的网格中。事实上，刚好落到网格的概率为0。所以从原来的DOA估计模型到稀疏恢复的模型中进行了zeroth-order approximation，即将真实的DOA近似到最近的网格中。零阶近似本质上是由于导向矢量的失配而产生的，再经过了信号能量的放大，进入到了噪声项中。所以信号能量越大，零阶近似的误差就越大。后续在BPDN的优化中，约束项需要同时考虑噪声的误差和零阶近似的误差。

单快拍$\mathbf{y}(t)$的稀疏表示意味着我们能够直接使用稀疏恢复技术来进行DOA估计。然而实际上，DOA估计得到的是多块拍数据$\mathbf{Y}$，称之为multiple measurement vectors (MMVs)，MMV的稀疏表征为（先假设没有噪声）
$$
\mathbf{Y} = \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X}
$$
可想而知，$\mathbf{X}$中的每一个列都共享相同的稀疏结构，称为这些列joint sparsity，而$\mathbf{X}$是行稀疏矩阵（row sparse）。所以显然MMV的稀疏恢复问题可以写为
$$
\min \| \mathbf{X} \|_{2, 0}, \;\; \text{s.t.} \; \mathbf{Y} = \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X}
$$
其中，$\| \mathbf{X} \|_{2, 0}$代表$\mathbf{X}$中非零行的数量，数学表示为
$$
\| \mathbf{X} \|_{2, 0} = \# \{n : \| \mathbf{X}_{n, :} \|_2 > 0\}
$$
同样进行凸松弛，将零范数松弛到一范数，并考虑到量测噪声和网格误差问题，稀疏表征模型变为了
$$
\mathbf{Y} = \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X} + \mathbf{E}
$$
其中，$\mathbf{E}$中不光包含了噪声产生的误差，也包含了零阶近似产生的误差，由此，BPDN问题可写为
$$
\min \| \mathbf{X} \|_{2, 1}, \;\; \text{s.t.} \; \| \mathbf{Y} - \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X} \|_{F} < \eta
$$
关于$\eta$如何选择。我们依旧选择Discrepancy principle准则，每次的观测$\mathbf{Y}$中，希望找到最紧贴的误差上界（包含噪声和网格误差）。对于噪声来说，所以要找到关于$\| \mathbf{E} \|_F$的分布情况
$$
\| \mathbf{E} \|_F \overset{\text{asym.}}{\sim} \frac{\sigma^2}{2} \chi^2(2ML) = \mathrm{Gamma}(\frac{2M L}{2}, 2\frac{\sigma^2}{2})
$$
由此找到一个合适的上界，使其在大部分情况下（0.999的概率）都小于这个上界（稍微紧就会有多峰），并且这个上界也不会超过一般情况太多（稍微松没什么问题，太松结果就会过于稀疏）。

### $\ell_1$-SVD method

参考：Sparse Methods for Direction-of-Arrival Estimation

A sparse signal reconstruction perspective for source localization with sensor arrays（这个还没认真看）

MMV稀疏恢复的BPDN问题，虽然是凸优化，直接可解，但是所需要的优化的变量为整个稀疏矩阵$\mathbf{X} \in \mathbb{C}^{\bar{N} \times L}$，十分巨大。启发于子空间类算法，特别是WSF算法，我们可以将观测$\mathbf{Y}$换成另外一种形式，且仍然保持相同的子空间，同时也可以利用噪声是白的这一先验信息除去一部分噪声。

当$L > K$时，将$\mathbf{Y}$进行特征值分解
$$
\mathbf{Y} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^H
$$
其中，$\mathbf{U}$为$\mathbf{Y}$列空间的基，$\mathbf{\Sigma}$为各个基在$\mathbf{Y}$列空间中所占的权重，我们仅保留$K$个基（即信号子空间），并且保留权重，即
$$
\mathbf{Y}_{\mathrm{SV}} = \mathbf{U} \mathbf{\Sigma}_{:, 1 : K}
$$
即只保留加权信号子空间$\mathbf{U} \mathbf{\Sigma}$的前$K$列（最大$K$个奇异值所对应的左奇异向量），在数学上，其还等于
$$
\begin{aligned}
\mathbf{Y}_{\mathrm{SV}} &= \mathbf{Y} \mathbf{V}_{:, 1 : K} \\
&= \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X}\mathbf{V}_{:, 1 : K} + \mathbf{E}\mathbf{V}_{:, 1 : K} \\
&= \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{X}_{\mathrm{SV}} + \mathbf{E}_{\mathrm{SV}}
\end{aligned}
$$
可以看到，$\mathbf{Y}_{\mathrm{SV}}$仍然有稀疏表征的结构，可以进行稀疏恢复，唯一需要注意的是噪声$\mathbf{E}_{\mathrm{SV}}$。由于$\mathbf{E}$中的每一个元素都是独立同分布的高斯随机变量，所以经过相互正交的$\mathbf{V}_{:, 1 : K}$的列之后，仍然是独立同分布的高斯随机变量，所以
$$
\| \mathbf{E}_{\mathrm{SV}} \|_F^2 \overset{\text{asym.}}{\sim} \frac{\sigma^2}{2} \chi^2(2MK) = \mathrm{Gamma}(\frac{2MK}{2}, 2\frac{\sigma^2}{2})
$$
然而，实际上SVD分解后的$\mathbf{V}$是与噪声$\mathbf{E}$有关的，在信噪比高时，$\mathbf{E}$在SVD中所占的成分不大，所以可以近似认为$\mathbf{V}$与噪声$\mathbf{E}$是无关的。但是随着信噪比的降低，$\mathbf{E}$在SVD中的占比越来越大，$\| \mathbf{E}_{\mathrm{SV}} \|_F^2$的分布就越来越不能按着原先的来了，具体如何选择还需要讨论。

与信号能量呈正相关的零阶近似误差也需要考虑。

### $\ell_1$-SRACV method

回想COMET算法，将估计协方差矩阵$\hat{\mathbf{R}} = \mathbf{R} + \Delta\mathbf{R}$ 作为观测，通过$\mathbf{R}$关于各种参数的信号模型，并且考虑$\Delta \mathbf{R}$的分布，对DOA进行多维搜索来完成估计。SRACV（Sparse Representation of Array Covariance vectors）通过利用$\mathbf{R}$的稀疏性来进行DOA（就像$\ell_1$方法用$\mathbf{Y}$的稀疏性来进行DOA估计）
$$
\begin{aligned}
\mathbf{R} &= \mathbf{A} \mathbf{P} \mathbf{A}^H + \sigma^2 \mathbf{I}_M \\
&= \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{B} + \sigma^2 \mathbf{I}_M
\end{aligned}
$$
由此，一个很自然的稀疏恢复优化问题为
$$
\min \| \mathbf{B} \|_{2, 0}, \;\; \mathrm{s.t.} \; \mathbf{R} = \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{B} + \sigma^2 \mathbf{I}_M
$$
考虑到实际中的实现，以及只能获得估计协方差矩阵$\hat{\mathbf{R}}$，不知道方差$\sigma^2$，需要将

- $\ell_0$范数松弛到$\ell_1$范数；
- 使用$\hat{\mathbf{R}}$转化为BPDN问题；
- 用估计方差$\hat{\sigma}^2$来代替$\sigma^2$。

所以，粗略的转换为如下的BPDN问题
$$
\min \| \mathbf{B} \|_{2, 1}, \;\; \mathrm{s.t.} \; \| \hat{\mathbf{R}} - \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{B} -\hat{\sigma}^2 \mathbf{I}_M\|_{F} \le \eta
$$
其中，$\hat{\sigma}^2$可由最小的$M - K$个特征值的均值来估计。$\eta$同时包含了三项误差：$\hat{\mathbf{R}}$的误差；网格近似误差；方差估计误差。若考虑到了$\Delta\mathbf{R}$的噪声分布，即
$$
\operatorname{vec}(\Delta \mathbf{R}) \overset{\text{asym.}}{\sim} \mathcal{CN}(\mathbf{0}, \frac{1}{N} \mathbf{R}^T \otimes \mathbf{R})
$$
同时使用估计的$\hat{\mathbf{W}} = N \mathbf{R}^{-T} \otimes \mathbf{R}^{-1}$作为LS的权重，由此约束项变为了
$$
\begin{aligned}
& \left\| \hat{\mathbf{W}}^{\frac{1}{2}} \left[\operatorname{vec} (\hat{\mathbf{R}} - \hat{\sigma}^2 \mathbf{I}_M) - \operatorname{vec} ( \mathbf{A}(\bar{\boldsymbol{\theta}}) \mathbf{B} )\right] \right\|_{2}  \\
=&  \left\| \hat{\mathbf{W}}^{\frac{1}{2}} \left[  \operatorname{vec} (\hat{\mathbf{R}} - \hat{\sigma}^2 \mathbf{I}_M) - (\mathbf{I}_{M} \otimes \mathbf{A}(\bar{\boldsymbol{\theta}}))  \operatorname{vec} ( \mathbf{B} )\right]  \right\|_{2}  \\
=&  \left\|  \mathbf{y} - \mathbf{\Psi} \operatorname{vec} ( \mathbf{B} )  \right\|_2 \le \eta
\end{aligned}
$$
其中
$$
\mathbf{y} = \hat{\mathbf{W}}^{\frac{1}{2}}  \operatorname{vec} (\hat{\mathbf{R}} - \hat{\sigma}^2 \mathbf{I}_M) \\ 
\mathbf{\Psi} = \hat{\mathbf{W}}^{\frac{1}{2}}(\mathbf{I}_{M} \otimes \mathbf{A}(\bar{\boldsymbol{\theta}}))
$$
由于$\mathbf{\Psi}$中包含了权重矩阵，所以其每个列都没有相同模值，实际的操作中需要进行归一化。

关于$\eta$的选择，需要考虑到之前所说的三种误差：$\hat{\mathbf{R}}$的误差；网格近似误差；方差估计误差。其中，第一种误差可以由$\mathbf{y} - \mathbf{\Psi} \operatorname{vec} ( \mathbf{B} )$的分布给出
$$
\mathbf{y} - \mathbf{\Psi} \operatorname{vec} ( \mathbf{B} ) \overset{\text{asym.}}{\sim} \mathcal{CN}(\mathbf{0}, \mathbf{I}_{M^2})
$$
所以
$$
\left\|  \mathbf{y} - \mathbf{\Psi} \operatorname{vec} ( \mathbf{B} )  \right\|_2^2 \overset{\text{asym.}}{\sim} \frac{1}{2}\chi^2(2 M^2)
$$
$\ell_1$-SRACV相比于COMET方法，只多做了将估计方差$\hat{\sigma}^2$作为方差的近似。其他的EXIP，和估计权重都一样。

$\eta$的选择很关键，现在都仅仅单独考虑了噪声的误差，其他的关于信号能量的网格误差；$\ell_1$-SVD在信噪比低时的误差；$\ell_1$-SRACV中关于方差的误差（不过这个误差应该不重要），都没有考虑。

