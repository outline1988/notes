### Least Squares and Solvers

有时候我们往往需要从一段的序列来获得某些参数的值，我们获取序列的数学建模如下
$$
\mathbf{r} = \mathbf{f}(\mathbf{x}) + \mathbf{w}
$$
其中，$\mathbf{r}$是观测向量，也即我们所获取的含有目标参数信息的序列；$\mathbf{x}$表示待估计的参数向量（也就是说要从序列中获取的参数不知一个）；$\mathbf{w}$表示噪声向量。

这里我们使用构造代价函数的方式来找到这些参数。观测向量$\mathbf{r}$是使用特定参数值向量生成序列后加上随机噪声而形成的。所以抛去噪声这个随机的因素，我们使用相同的参数值向量是完全可以生成与观测向量完全一致的序列的。

所以，最暴力的方法就是遍历所有可能的参数值向量，将每组参数值向量都用同样的方式生成与观测向量同样长度的序列，将该序列与观测向量比较，与之最相似的序列所对应的参数值向量就是我们想要的答案。

那么如何认为遍历到的某一参数值向量$\tilde{\mathbf{x}}$所生成的模拟序列$\tilde{\mathbf{r}}$，就是最接近观测向量$\mathbf{r}$的呢？答案就是使用代价函数，代价函数通过距离来评估两个序列的相似程度，而这里所提到的距离就是最小二乘距离，即
$$
\begin{gathered}
 \\
J(\tilde{\mathbf{x}})=\left(\mathbf{r}-\mathbf{f}(\tilde{\mathbf{x}})\right)^{T}\left(\mathbf{r}-\mathbf{f}(\tilde{\mathbf{x}})\right)=\|\mathbf{r}-\mathbf{f}(\tilde{\mathbf{x}})\|_{2}^{2}=\sum_{n = 1}^{N}\left(r[n]-f_{n}(\tilde{\mathbf{x}})\right)^{2} \\
\; 
\end{gathered}
$$
我们可以在参数搜索空间中找到所有可能的参数值向量，来计算对应代价函数，从而找到最小值。

也可以使用**梯度下降法**，也即使用迭代的方式，使用梯度的方向是指向变化速率最大的方向的性质，来找到代价函数的最低点，数学表示如下
$$
\hat{\mathbf{x}}(k+1)=\hat{\mathbf{x}}(k)-\mu J'(\hat{\mathbf{x}}(k))
$$
给定一个初始值$\hat{\mathbf{x}}(0)$，然后对其求梯度（一维就是求导，符号就是方向），从而找到此点的下降最快的方向（梯度方向的反方向），再根据具体的要求选择步长$\mu$，迭代一定次数之后，就能收敛到代价函数的最低值了。



任意一个正弦信号，都可以表示为两个同频率的正弦信号之和，以这样的两个同频率信号为一组，其组数是无穷的，通过几何表示可以直观的证明。

### Adaptive Filter

看图像作业去。。。

### Global Thresholding

**Otsu's Method**
Otsu的方法在于对已经分割的图像，其使用特定的评价指标来评价分割后的两个区域$G_1$和$G_2$的距离程度，通过暴力地列举所有可能的阈值$T$，找到距离程度最大的阈值，即为Otsu的最终结果。

Otsu方法的核心就是如何评估两个区域的距离程度，其使用下式来评估
$$
\sigma_B = P_1(m_1 - m_G)^2 + P_2(m_2 - m_G)^2
$$
其同样可以写成
$$
\sigma_B = P_1 P_2 (m_1 - m_2)^2 = \frac{(m_G P_1 - m)^2}{P_1(1 - P_1)}
$$
其本质就是两个区域对于整体的方差的加权和。

以下是具体步骤：遍历所有可能的阈值$T = k$，其中$k$的范围为$[0, L - 2]$。对于每个$k$，将其作为阈值分割图像，从而分出两个区域$G_1$和$G_2$，分别计算第一个区域的均值$m_1$和区域出现的概率$P_1$，从而计算出对于该阈值$T = k$的区域距离程度。当算出所有可能的阈值的距离程度后，找到最大的$\sigma_B$即可。

