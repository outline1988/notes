### 从匹配滤波的视角上看协方差矩阵估计

DOA估计的多快拍信号模型为

$$
\mathbf{X} = \mathbf{A} \mathbf{S} + \mathbf{N}  
$$

  

确定性模型的最大似然估计量（DML）为

$$
J = \operatorname{tr}(\hat{\mathbf{R}} \mathbf{P}_{\mathbf{A}})  
$$

  

其中

$$
\begin{aligned} \hat{\mathbf{R}} &= \mathbf{X}\mathbf{X}^H \\ &= \begin{bmatrix} \mathbf{x}_1^T \mathbf{x}_1^* & \cdots & \mathbf{x}_1^T \mathbf{x}_M^* \\  \mathbf{x}_2^T \mathbf{x}_1^* & \cdots & \mathbf{x}_2^T \mathbf{x}_M^* \\ \vdots & & \vdots \\ \mathbf{x}_M^T \mathbf{x}_1^* & \cdots & \mathbf{x}_M^T \mathbf{x}_M^* \\ \end{bmatrix} \end{aligned}  
$$

  

为协方差矩阵的估计量的N倍（这里为了方便，不考虑前面的稀疏N）。$\mathbf{x}_m$为第$m$个阵元所接收的$N$个快拍的列向量。可以发现，$\hat{\mathbf{R}}$中包含着不同阵元与不同阵元之间的相关。

匹配滤波器通过匹配的波形将N个快拍的接收数据的能量积累到一个快拍，从而提高那个单快排的信噪比，易于后续的检测与估计。那么，是否能从匹配滤波的能量积累的角度来考虑DOA估计的过程呢？

无论是DML还是MUSIC，其都依赖于协方差矩阵的估计$\hat{\mathbf{R}}$，可以认为，其就是充分统计量，包含了估计DOA的所有信息。

考虑$\hat{\mathbf{R}}$中的某个元素

$$
[ \hat{\mathbf{R}} ]_{ij} = \mathbf{x}^T_{i}\mathbf{x}_{j}^*  
$$

  

其中

$$
\begin{aligned} \mathbf{x}_m^T = \mathbf{X}_{m, :} = \mathbf{a}^T_m(\boldsymbol{\theta}) \mathbf{S} + \mathbf{n}_m^T \end{aligned}  
$$

  

这是因为

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_{1}^T \\ \vdots \\ \mathbf{x}_{M}^T \end{bmatrix} = \begin{bmatrix} \mathbf{a}_{1}^T(\boldsymbol{\theta}) \\ \vdots \\ \mathbf{a}_{M}^T(\boldsymbol{\theta}) \end{bmatrix}\mathbf{S} + \begin{bmatrix} \mathbf{n}_{1}^T \\ \vdots \\ \mathbf{n}_{M}^T \end{bmatrix}  
$$

  

$$
\mathbf{a}_m(\boldsymbol{\theta}) = \mathbf{A}_{m, :}^T  
$$

  

由此

$$
\begin{aligned} \mathbf{x}^T_{i}\mathbf{x}_{j}^* &= \left[\mathbf{a}^T_i(\boldsymbol{\theta}) \mathbf{S} + \mathbf{n}_i^T\right] \left[\mathbf{S}^H \mathbf{a}^*_j(\boldsymbol{\theta}) + \mathbf{n}_j^*\right] \\ &= \mathbf{a}^T_i(\boldsymbol{\theta}) \mathbf{S} \mathbf{S}^H \mathbf{a}^*_j(\boldsymbol{\theta}) + \mathbf{a}^T_i(\boldsymbol{\theta}) \mathbf{S} \mathbf{n}_j^* + \mathbf{n}_i^T \mathbf{S}^H \mathbf{a}^*_j(\boldsymbol{\theta}) + \mathbf{n}_i^T \mathbf{n}_j^* \end{aligned}  
$$

  

若只考虑无随机的第一项

$$
\begin{aligned} \mathbf{a}^T_i(\boldsymbol{\theta}) \mathbf{S} \mathbf{S}^H \mathbf{a}^*_j(\boldsymbol{\theta}) &= \sum\limits_{k, l}^{K} a_i(\theta_k) a_j^* (\theta_l) \mathbf{s}^T_k \mathbf{s}_l^* \end{aligned}  
$$

$$
[ \hat{\mathbf{R}} ]_{ij} = \mathbf{y}^T_{i}\mathbf{y}_{j}^*  

$$


也即，估计协方差矩阵的某一个元素$[ \hat{\mathbf{R}} ]_{ij}$的无随机项，是K个不同信号在N个快拍的相关在不同权重下的加权和。

在考虑最后一项，$\mathbf{n}_i^T \mathbf{n}_j^*$，其只有当$i = j$时，能量才能被积累，其他情况下能量也很低。

除了第一项和最后一项$i = j$的情况，中间两项的能量也不能被积累。

也就是说，协方差矩阵的相关使得第一项和最后一项$i = j$的情况下得到了突出。而DOA估计也应该是在突出的信息上再进行的。

所以说，估计协方差矩阵中包含了匹配滤波中能量积累的过程，但是积累出的能量并非是如同时延估计中一样直接能够用来检测或者估计，而是需要进一步处理，即对估计协方差矩阵进行处理。

### 多源搜索的单维近似

参考：Three More Decades in Array Signal Processing Research: An optimization and structure exploitation perspective

Sensor array processing based on subspace fitting

一个subspace fitting问题的基本形式如下
$$
\hat{\mathbf{A}}, \hat{\mathbf{T}} = \arg \min_{\mathbf{A}, \mathbf{T}} \left\| \mathbf{M} - \mathbf{A} \mathbf{T} \right\|_F^2
$$
对于一个固定的$\mathbf{A}$，该表达式实际上在计算出$\mathbf{M}$和$\mathbf{A}$列空间的距离。所以，对于多个可选择的$\mathbf{A}$，该优化问题旨在挑选出离$\mathbf{M}$列空间中最近的$\mathbf{A}$。选择不同的$\mathbf{M}$，即可对应不同的算法。两种比较常见的选择为$\mathbf{M} = \mathbf{X}$和$\mathbf{M} = \hat{\mathbf{U}}_\mathrm{s}$，前者对应DML算法，后者对应WSF算法。

根据文章，多维搜索的问题由于计算复杂度极其之高，通常使用approximation或者relaxation等方式在以性能为代价的情况下进行简化，这里讨论single-source approximation，也即多维的优化问题变为了
$$
{}^N\!\arg \min_{\mathbf{a}, \mathbf{v}} \left\| \mathbf{M} - \mathbf{a} \mathbf{v}^T \right\|_F^2
$$
其中，${}^N\!\arg \min$表示取最小的$N$个值。

对于DML的single-source approximation，直接变为了CBF的形式
$$
\begin{aligned}
\operatorname{tr}(\mathbf{P}_{\mathbf{a}} \mathbf{X}\mathbf{X}^H) &= \sum\limits_{t = 1}^N \mathbf{x}^H(t) \mathbf{P}_{\mathbf{a}} \mathbf{x}(t) \\
&= \sum\limits_{t = 1}^{N} \left| \mathbf{a}^H\mathbf{x}(t) \right|^2
\end{aligned}
$$
对于WSF的single-source approximation，直接变为了MUSIC的形式
$$
\begin{aligned}
\operatorname{tr}(\mathbf{P}_{\mathbf{a}} \hat{\mathbf{U}}_\mathrm{s}\hat{\mathbf{U}}_\mathrm{s}^H) &= \sum\limits_{k = 1}^K \hat{\mathbf{u}}_{\mathrm{s}}^H(k) \mathbf{P}_{\mathbf{a}} \hat{\mathbf{u}}_{\mathrm{s}}(k) \\
&= \sum\limits_{k = 1}^{K} \left| \mathbf{a}^H\hat{\mathbf{u}}_{\mathrm{s}}(k) \right|^2 \\
&= \mathbf{a}^H \hat{\mathbf{U}}_{\mathrm{s}} \hat{\mathbf{U}}_{\mathrm{s}}^H \mathbf{a}
\end{aligned}
$$
那么众所周知，CBF和MUSIC的性能差异很大，MUSIC拥有很好的超分辨能力，而CBF不具有。而他们的源头DML和WSF的算法效果确实差不多的，甚至DML因为是最大似然估计，被认为是最优的benchmark，那么究竟是什么原因导致了single-source approximation之后的算法差异这么大呢？

我认为，是CBF没有利用上噪声是白的这一先验信息。而MUSIC利用了噪声是白的这一先验信息。而源头的DML和WSF，都利用了噪声是白的这一先验信息。

很奇怪，在仿真中，对于$\mathbf{M}$的不同选择，对于单源近似后的影响很大，即使是MUSIC和W-MUSIC，到底是怎么回事呢？

关键的式子是$\mathbf{A}^H \mathbf{U}_{\mathrm{n}} = \mathbf{0}$，无论怎么选择$\mathbf{M}$，一定要使得搜索的时候该式成立，这样才有超分辨的效果。只有当$\mathbf{M} = \mathbf{U}_{\mathrm{s}}$时，才会在搜索的时候蕴含$\mathbf{A}^H \mathbf{U}_{\mathrm{n}} = \mathbf{0}$的效果，此时才是最好的，那么如何从单源近似的角度来理解只有当信号子空间的情况时，效果才是最好的呢，其他的都不行，即使是同一子空间，但是不是正交的矩阵。

### Covariance Matching（COMET）

参考：Covariance Matching Estimation Techniques for Array Signal Processing Applications（虽然这里列为参考，但是写的依托）

On reparametrization of loss functions used in estimation and the invariance principle（主要是EXIP的概念，我称之为分步方法的理论支撑）

COMET方法来源于对最大似然函数使用EXIP，类似于最大似然的不变性。将同一优化函数用不同的参数表示出来，得到新参数的最大似然估计量，再映射回原来参数。具体参考第二篇文章。

Covariance matching estimation techniques（COMET）首先使用$N$个快拍对协方差矩阵进行估计
$$
\hat{\mathbf{R}} = \frac{1}{N} \sum\limits_{t = 1}^{N} \mathbf{y}(t)\mathbf{y}^H(t)
$$
将其作为观测数据，进行之后的估计，即信号模型为
$$
\hat{\mathbf{R}} = \mathbf{R} + \Delta \mathbf{R}
$$
其中，$\hat{\mathbf{R}}$是观测，$\Delta \mathbf{R}$是噪声以及$\mathbf{R}$是真值，$\mathbf{R}$的表达式如下
$$
\mathbf{R} = \mathbf{A} \mathbf{P} \mathbf{A}^H + \sigma^2 \mathbf{I}
$$
若噪声方差未知，则待估计的参数为
$$
\boldsymbol{\xi} = \begin{bmatrix}
\theta_1 & \cdots & \theta_K & P_{11} & \cdots & P_{KK} & \sigma^2
\end{bmatrix}
$$
包括感兴趣的参数$\boldsymbol{\theta}$，以及nuisance参数为信号协方差矩阵$\mathbf{P}$的各个元素和方差$\sigma^2$。

$\mathbf{R}$中包含有线性参数和非线性参数，为了方便表示，将其向量化
$$
\begin{aligned}
\mathbf{r} &= \operatorname{vec}(\mathbf{R}) 
&= \operatorname{vec}(\mathbf{A} \mathbf{P} \mathbf{A}^H + \sigma^2 \mathbf{I}) \\
&= (\mathbf{A}^* \otimes \mathbf{A}) \operatorname{vec}(\mathbf{P}) + \operatorname{vec}(\mathbf{I}_M) \sigma^2 \\
&= \begin{bmatrix}
(\mathbf{A}^* \otimes \mathbf{A}) & \operatorname{vec}(\mathbf{I}_M)
\end{bmatrix}
\begin{bmatrix}
\operatorname{vec}(\mathbf{P}) \\
\sigma^2
\end{bmatrix} \\
&= \mathbf{\Phi}(\boldsymbol{\theta}) \boldsymbol{\alpha} \\
\end{aligned}
$$
其中，$\boldsymbol{\theta}$是非线性参数，$\boldsymbol{\alpha}$是线性参数。（这里$\mathbf{P}$内的元素实际上还满足一些约束关系，但是这里没有利用上，是否有问题呢？）

由此，我们使用最大似然估计（加权最小二乘）
$$
\begin{aligned}
J &= \left[ \operatorname{vec}(\hat{\mathbf{R}}) -  \operatorname{vec}({\mathbf{R}}) \right]^H \mathbf{W}\left[ \operatorname{vec}(\hat{\mathbf{R}}) -  \operatorname{vec}({\mathbf{R}}) \right]  \\
&= (\hat{\mathbf{r}} - \mathbf{r})^H \mathbf{W} (\hat{\mathbf{r}} - \mathbf{r})
\end{aligned}
$$
其中，$\mathbf{W}$为$\operatorname{vec}(\Delta{\mathbf{R}})$的协方差矩阵的逆（分步法CRB的传递）。根据文章，可知其表达式为
$$
\mathbf{W}^{-1} = \frac{1}{N} \mathbf{R}^T \otimes \mathbf{R}
$$

通常使用估计协方差矩阵来代替（说实话我不知道这样的代替有没有道理，但是反正论文是这么做的，可能$\mathbf{W}$只是一个权重，将其近似确定后，近似的误差小）
$$
\hat{\mathbf{W}}^{-1} = \frac{1}{N} \hat{\mathbf{R}} ^T \otimes \hat{\mathbf{R}}
$$


$J$可以继续化简为
$$
\begin{aligned}
\frac{1}{N}J &\approx (\hat{\mathbf{r}} - \mathbf{r})^H \hat{\mathbf{R}} ^{-T} \otimes \hat{\mathbf{R}}^{-1}  (\hat{\mathbf{r}} - \mathbf{r}) \\
&= \Delta{\mathbf{r}}^H (\hat{\mathbf{R}} ^{-T} \otimes \hat{\mathbf{R}} ^{-1}) \operatorname{vec}(\Delta \mathbf{R}) \\
&= \operatorname{vec}(\Delta \mathbf{R})^H \operatorname{vec}(\hat{\mathbf{R}} ^{-1} \Delta \mathbf{R}\hat{\mathbf{R}} ^{-1}) \\
&= \operatorname{tr}(\Delta \mathbf{R} \hat{\mathbf{R}} ^{-1} \Delta \mathbf{R} \hat{\mathbf{R}} ^{-1})
\end{aligned}
$$
可以继续转化为Frobenius范数的形式
$$
\begin{aligned}
\frac{1}{N} J &\approx \operatorname{tr}( \Delta \mathbf{R} \hat{\mathbf{R}} ^{-1} \Delta \mathbf{R} \hat{\mathbf{R}} ^{-\frac{1}{2}} \hat{\mathbf{R}} ^{-\frac{1}{2}} ) \\
&= \operatorname{tr}( \hat{\mathbf{R}}^{-\frac{1}{2}} \Delta \mathbf{R} \hat{\mathbf{R}}^{-\frac{1}{2}} \hat{\mathbf{R}}^{-\frac{1}{2}} \Delta \mathbf{R}\hat{\mathbf{R}}^{-\frac{1}{2}} ) \\
&= \left\| \hat{\mathbf{R}}^{-\frac{1}{2}} (\hat{\mathbf{R}} - \mathbf{R}) \hat{\mathbf{R}}^{-\frac{1}{2}}  \right\|_F^2
\end{aligned}
$$
具体的数学运算 ，可以参考Matrix_Notes2中关于Frobenius那一节。光从这个式子来理解，也是很好理解的，用已知信号模型的$\mathbf{R}$来适配观测$\hat{\mathbf{R}}$，因为要考虑到噪声，所以还需要在前后加上$\hat{\mathbf{R}}^{-\frac{1}{2}}$。

除了使用F范数将目标函数简化，还可以利用线性参数分离的方式来进行简化
$$
\begin{aligned}
\frac{1}{N} J &\approx (\hat{\mathbf{r}} - \mathbf{r})^H \hat{\mathbf{W}} (\hat{\mathbf{r}} - \mathbf{r}) \\
&= \left\| \hat{\mathbf{W}}^{\frac{1}{2}} (\hat{\mathbf{r}} - \mathbf{\Phi}(\boldsymbol{\theta}) \boldsymbol{\alpha} ) \right\|_2^2 \\
&=  \left\| \hat{\mathbf{r}}' - \mathbf{\Phi}'(\boldsymbol{\theta}) \boldsymbol{\alpha}  \right\|_2^2 \\
&= \hat{\mathbf{r}}'^H \mathbf{P}_{\mathbf{\Phi}'}^{\perp} \hat{\mathbf{r}}'
\end{aligned}
$$
该式可以继续简化
$$
\hat{\mathbf{r}}' = \hat{\mathbf{W}}^{\frac{1}{2}} \hat{\mathbf{r}}, \;\;\mathbf{\Phi}' = \hat{\mathbf{W}}^{\frac{1}{2}} \mathbf{\Phi}
$$

$$
\begin{aligned}
\mathbf{P}_{\mathbf{\Phi}'} &=  \mathbf{\Phi}'(\mathbf{\Phi}'^H\mathbf{\Phi}')^{-1}\mathbf{\Phi}'^H \\
&= \hat{\mathbf{W}}^{\frac{1}{2}} \mathbf{\Phi} \left(  \mathbf{\Phi}^H \hat{\mathbf{W}} \mathbf{\Phi} \right) \mathbf{\Phi}^H\hat{\mathbf{W}}^{\frac{1}{2}}
\end{aligned}
$$

由此，最小化原式相当于最大化
$$
\begin{aligned}
 \hat{\mathbf{r}}'^H \mathbf{P}_{\mathbf{\Phi}'} \hat{\mathbf{r}}' &=  \hat{\mathbf{r}}^H\hat{\mathbf{W}}^{\frac{1}{2}}\hat{\mathbf{W}}^{\frac{1}{2}} \mathbf{\Phi} \left(  \mathbf{\Phi}^H \hat{\mathbf{W}} \mathbf{\Phi} \right) \mathbf{\Phi}^H \hat{\mathbf{W}}^{\frac{1}{2}} \hat{\mathbf{W}}^{\frac{1}{2}} \hat{\mathbf{r}} \\
 &= \hat{\mathbf{r}}^H\hat{\mathbf{W}}\mathbf{\Phi} \left(  \mathbf{\Phi}^H \hat{\mathbf{W}} \mathbf{\Phi} \right) ^{-1}\mathbf{\Phi}^H\hat{\mathbf{W}} \hat{\mathbf{r}}
\end{aligned}
$$
其中
$$
\begin{aligned}
\hat{\mathbf{W}} \hat{\mathbf{r}} &= N (\hat{\mathbf{R}} ^{-T} \otimes \hat{\mathbf{R}}^{-1}) \operatorname{vec}(\hat{\mathbf{r}}) \\
&= N \operatorname{vec}( \hat{\mathbf{R}}^{-1}\hat{\mathbf{R}}\hat{\mathbf{R}}^{-1} ) \\
&= N \operatorname{vec}( \hat{\mathbf{R}}^{-1} )
\end{aligned}
$$
由此
$$
\begin{aligned}
\hat{\boldsymbol{\theta}} &= \arg \max\limits_{\boldsymbol{\theta}} \left[ \operatorname{vec}( \hat{\mathbf{R}}^{-1} )^H \mathbf{\Phi} \left(  \mathbf{\Phi}^H \hat{\mathbf{W}} \mathbf{\Phi} \right) ^{-1}\mathbf{\Phi}^H \operatorname{vec}( \hat{\mathbf{R}}^{-1} ) \right] \\
\end{aligned}
$$
其中
$$
\mathbf{\Phi} = \begin{bmatrix}
(\mathbf{A}^* \otimes \mathbf{A}) & \operatorname{vec}(\mathbf{I}_M)
\end{bmatrix}
$$
注意，这里将方差视为未知参数，从而得到的这个式子。

回顾整个过程，相比于SML，COMET进行了两次近似

1. 使用EXIP的思想，先最大似然估计得到估计协方差$\hat{\mathbf{R}}$，再将其作为量测进行最大似然估计，类似于分布估计的思路；
2. 分布估计的第二步，即将$\hat{\mathbf{R}}$作为量测时的最大似然估计的权重，使用$\hat{\mathbf{R}}$来进行固定（本来应该是随着角度一起搜索）。














