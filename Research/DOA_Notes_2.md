### Deterministic Maximum Likelihood

#### 直接从多块拍推导

将接收数据$X$进行拉伸运算，等价于$\left\{ \mathbf{x}(t) \right\}_{t = 1}^{N}$排列为列向量的形式
$$
\begin{aligned}
\mathbf{x} &= \operatorname{vec}(X) \\
&= \operatorname{vec}\left(AS + E\right) \\
&= \left( I \otimes A  \right) \operatorname{vec}(S) + \operatorname{vec}(E) \\
&= G \mathbf{s} + \mathbf{e} \\
\end{aligned}
$$
所以拉长后观测数据的对数似然函数为
$$
\ln p(\mathbf{x}; \boldsymbol{\xi}) = c - \frac{1}{\sigma^2} (\mathbf{x} - G \mathbf{s})^H (\mathbf{x} - G \mathbf{s})
$$
其中，噪声为方差$\sigma^2$的白噪声（已知）。待估计的参数为（确定性假设的波形拆分为了实部和虚部）
$$
\boldsymbol{\xi} = \begin{bmatrix}
\theta_1 & \cdots & \theta_K & \bar{\mathbf{s}}^T & \tilde{\mathbf{s}}^T
\end{bmatrix}^T
$$
观察数据模型啊，可知该模型包含了线性部分的参数$\bar{\mathbf{s}}$和$\tilde{\mathbf{s}}$，以及非线性部分的参数$\boldsymbol{\theta}$，所以可以采用参数分离的技巧，即线性参数的估计量可有非线性参数表示
$$
\hat{\mathbf{s}} = \left(\hat{G}^H \hat{G} \right)^{-1}\hat{G}^H \mathbf{x}
$$
由此最大化似然等价于最小化一下目标函数
$$
\begin{aligned}
J &= (\mathbf{x} - G \hat{\mathbf{s}})^H (\mathbf{x} - G \hat{\mathbf{s}}) \\
&= \left[\mathbf{x} - G \left(G^H G \right)^{-1}G^H \mathbf{x}\right]^H \left[\mathbf{x} - G \left(G^H G \right)^{-1}G^H \mathbf{x}\right] \\
&= \left(P_G^{\perp} \mathbf{x}\right)^H\left(P_G^{\perp} \mathbf{x}\right) \\
&= \mathbf{x}^H P_G^{\perp} \mathbf{x} \\
&= \sum\limits_{t = 1}^N \mathbf{x}^H(t) P_A^{\perp} \mathbf{x}(t)
\end{aligned}
$$
同样，也可以最大化$\sum\limits_{t = 1}^N \mathbf{x}^H(t) P_A \mathbf{x}(t)$或者$\operatorname{tr} \left( P_A \hat{R}_{xx}  \right)$，注意前者的表达式可以理解为$\mathbf{x}(t)$向量投影到$A$列空间后向量模长的平方。

由于非线性的影响，目标函数具有复杂的多峰形状（非凸）。在实际的操作中，我们以各种各样的方式多维搜索$\boldsymbol\theta$（比如网格搜索），来使得对应的目标函数进行最优化。

#### 从单快拍推广到多快拍

确定性模型单快拍接收数据为
$$
\mathbf{x}(t) = \mathbf{A} \mathbf{s}(t) + \mathbf{e}(t) \sim \mathcal{CN}(\mathbf{A} \mathbf{s}(t), \sigma^2 \mathbf{I})
$$
未知参数为角度$\theta_1, \cdots, \theta_K$，以及该时刻的信号波形$s_1(t), \cdots, s_K(t)$。故单快拍接收数据的似然函数
$$
p(\mathbf{x}(t)) = \frac{1}{\pi^M \det (\sigma^2 \mathbf{I})} \exp\left( -\frac{1}{\sigma^2}[\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)]^H [\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)] \right)
$$
对数似然为
$$
\ln p(\mathbf{x}(t)) = -M \ln\pi - 2M \ln \sigma -\frac{1}{\sigma^2}[\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)]^H [\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)]
$$
由于多个快拍中，不同快拍是独立的，所以对不同时刻的对数似然函数求和就是所有时刻的对数似然，再舍弃无关项，最终整理为目标函数
$$
J = \sum\limits_{t = 1}^{N} [\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)]^H [\mathbf{x}(t) - \mathbf{A} \mathbf{s}(t)]
$$
有关于DML的性能分析，可以参考文章：Maximum likelihood methods for direction-of-arrival estimation

### Stochastic Maximum Likelihood

现在我们考虑随机性最大似然估计，其接收数据的模型与确定性拥有相同的形式，不同来源于对于信号的假设是随机的，且在空域中是平稳的
$$
\mathbf{x}(t) = \mathbf{A} \mathbf{s}(t) + \mathbf{e}(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{R})
$$
其中，$\mathbf{s}(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{P})$，即不同的信号之间是平稳的，具有某个协方差$\mathbf{P}$，不同时刻的信号是独立的。

未知参数为角度$\theta_1, \cdots, \theta_K$，以及该时刻的信号协方差的各个元素。故单快拍接收数据的似然函数
$$
p(\mathbf{x}(t)) = \frac{1}{\pi^M \det (\mathbf{R})} \exp\left[ -\mathbf{x}(t)^H \mathbf{R}^{-1} \mathbf{x}(t) \right]
$$
对数似然为
$$
\begin{aligned}
\ln p(\mathbf{x}(t)) &= -M \ln\pi - \ln \det(\mathbf{R}) - \mathbf{x}(t)^H \mathbf{R}^{-1} \mathbf{x}(t) \\
&= -M \ln\pi - \ln \det(\mathbf{R}) - \operatorname{tr}\left[\mathbf{x}(t)\mathbf{x}(t)^H \mathbf{R}^{-1}\right]
\end{aligned}
$$
由于多个快拍中，不同快拍是独立的，所以对不同时刻的对数似然函数求和就是所有时刻的对数似然，再舍弃无关项，最终整理为目标函数
$$
J =  \ln \det(\mathbf{R}) +  \operatorname{tr}\left(\hat{\mathbf{R}} \mathbf{R}^{-1}\right)
$$


### Expectation-Maximization

参考：Theory and Use of the EM Algorithm. 

Parameter estimation of superimposed signals using the EM algorithm. 

Maximum-likelihood narrow-band direction finding and the EM algorithm. 

对于DOA估计的最大似然求解问题，Maximum-Likelihood DOA Estimation by Data-Supported Grid Search这篇文章将求解方法按照计算复杂度从低到高分为四类：大样本高信噪比近似方法、局部搜索方法、全局搜索方法以及网格搜索法。由于近似方法得到的估计量本质上不属于最大似然估计的范畴了，所以本节不考虑。局部搜索方法即使用优化理论的方法以似然函数作为优化的目标函数，常见的有EM、梯度类、ADMM等方法，本节考虑EM算法。

EM算法的步骤和基本原理参考Theory and Use of the EM Algorithm，简单来说，EM算法使用MM（Maximization-Maximization）准则，通过对似然函数在某次迭代参数值的下界进行优化，使得似然函数在迭代的参数值下具有单调不降的特性，由此收敛到似然函数的稳定点。

可以将EM算法具体运用在多信号叠加参数估计的问题上，相应的信号模型为
$$
\mathbf{y}(t) = \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) + \mathbf{n}(t)
$$
其中，$\mathbf{y}(t)$可以为任意维的列向量（当然也包括标量），$\boldsymbol{\theta}$为该模型下待估计的所有参数，$\mathbf{s}_k(t; \boldsymbol{\theta})$表示从全局参数$\boldsymbol{\theta}$到第$k$个时域信号的映射，噪声为高斯噪声$\mathbf{n}(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{Q})$。直观来说，观测的数据由多个信号以及噪声叠加而成，其中每个信号都可以由一个全局的参数$\boldsymbol{\theta}$以各自不同的映射$\mathbf{s}_k(t; \boldsymbol{\theta})$来产生。如果这个模型满足控制每个信号源的参数独立，不会相互影响，也即将参数$\boldsymbol{\theta}$分开为$K$个互不重叠的参数$\boldsymbol{\theta}_k$，第$k$个信号源只由$\boldsymbol{\theta}_k$独立产生，那么就变为了论文中的模型
$$
\mathbf{y}(t) = \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}_k) + \mathbf{n}(t)
$$
暂时先考虑第一种具有全局参数的模型，似然函数为
$$
\ln p(\mathbf{y}; \boldsymbol{\theta}) = c_1 - \sum\limits_{t = 1}^{N} \left[ \mathbf{y}(t) -  \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\mathbf{y}(t) -  \sum\limits_{k = 1}^K \mathbf{s}_k(t; \boldsymbol{\theta}) \right]
$$
根据EM算法，我们要指定一个隐变量（complete data）$\mathbf{x}$，其与参数$\boldsymbol{\theta}$，观测数据$\mathbf{y}$应该满足$\boldsymbol{\theta} \rightarrow \mathbf{x} \rightarrow \mathbf{y}$的Markov链的关系，也即$\mathbf{y}$只取决于$\mathbf{x}$而不取决于参数$\boldsymbol{\theta}$。在多信号叠加的模型下，我们指定隐变量为每个单独的信号源外加一个噪声
$$
\mathbf{x}(t) = \begin{bmatrix}
\mathbf{x}_1(t) \\
\vdots \\
\mathbf{x}_K(t)
\end{bmatrix} \\
\mathbf{x}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}) + \mathbf{n}_k(t)
$$

$$
\mathbf{y}(t) = \begin{bmatrix}
\mathbf{I} & \cdots & \mathbf{I}
\end{bmatrix}
\mathbf{x}(t)  = \mathbf{H} \mathbf{x}(t)
$$

其中，假设每个子噪声$\mathbf{n}_k(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{Q}_k)$为相互独立，则可以指定每个子噪声的方差$\mathbf{Q}_k = \beta_k \mathbf{Q}$，满足$\sum\limits_{k = 1}^{K} \beta_k = 1$。由此隐变量的似然函数为
$$
\begin{aligned}
\ln p(\mathbf{x}; \boldsymbol{\theta}) &= c_2 - \sum\limits_{t = 1}^{N} \left[ \mathbf{x}(t) -  \mathbf{s}(t; \boldsymbol{\theta}) \right]^H \mathbf{\Lambda}^{-1} \left[\mathbf{x}(t) -  \mathbf{s}(t; \boldsymbol{\theta}) \right] \\
&= c_2 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \mathbf{x}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\mathbf{x}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right] \\
&= c_3 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[
-\mathbf{x}^H_k(t) \mathbf{Q}^{-1}\mathbf{s}_k(t; \boldsymbol{\theta}) - \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{x}_k(t) + \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{s}_k(t; \boldsymbol{\theta})
\right]
\end{aligned}
$$

$$
\mathbf{\Lambda} = \begin{bmatrix}
\mathbf{Q}_1 & & \\
& & \ddots & \\
& & & \mathbf{Q}_K
\end{bmatrix}
$$

在EM算法进行E-step时，需要计算隐变量似然函数的条件期望（假设参数迭代到了$\boldsymbol{\theta}^{(n)}$）
$$
\begin{aligned}
& E_{\mathbf{x} \mid \mathbf{y}} \left[\ln p(\mathbf{x}; \boldsymbol{\theta})  ; \boldsymbol{\theta}^{(n)} \right] \\
&= c_4 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[
-\hat{\mathbf{x}}^H_k(t) \mathbf{Q}^{-1}\mathbf{s}_k(t; \boldsymbol{\theta}) - \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \hat{\mathbf{x}}_k(t) + \mathbf{s}_k^H(t; \boldsymbol{\theta}) \mathbf{Q}^{-1} \mathbf{s}_k(t; \boldsymbol{\theta}) \right] \\
&= c_5 - \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]
\end{aligned}
$$
其中，根据高斯随机变量的条件均值公式$E[\mathbf{x} \mid \mathbf{y}] = E[\mathbf{x}] + C_{xy}C_{yy}^{-1} \left[ \mathbf{y} - E[\mathbf{y}] \right]$
$$
\begin{aligned}
\hat{\mathbf{x}}(t) &= E_{\mathbf{x} \mid \mathbf{y}} \left[\mathbf{x}(t) ; \boldsymbol{\theta}^{(n)} \right] \\
&= \mathbf{s}_k(t; \boldsymbol{\theta}^{(n)}) + \mathbf{\Lambda} \mathbf{H}^T \left[ \mathbf{H} \mathbf{\Lambda} \mathbf{H} \right]^{-1} \left[ \mathbf{y}(t) - \mathbf{H} \mathbf{s}(t; \boldsymbol{\theta}^{(n)} ) \right]
\end{aligned}
$$
展开后再将向量内的元素拆开可以得到更加简洁的表达式
$$
\hat{\mathbf{x}}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}^{(n)}) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K} \mathbf{s}_l(t; \boldsymbol{\theta}^{(n)}) \right]
$$
由此即可完成E-step的计算。

在M-step中，我们需要优化的目标函数为$E_{\mathbf{x} \mid \mathbf{y}} \left[\ln p(\mathbf{x}; \boldsymbol{\theta})  ; \boldsymbol{\theta}^{(n)} \right]$，由E-step可知，在高斯噪声与隐变量与观测变量是线性关系的假设下，其相当于在参数为$\boldsymbol{\theta}^{(n)}$时计算出了隐变量（complete data）的观测数据$\hat{\mathbf{x}}$，然后再求出隐变量模型的最大似然解，隐变量模型为
$$
\mathbf{x}(t) = \begin{bmatrix}
\mathbf{s}_1(t; \boldsymbol{\theta})  \\
\vdots \\
\mathbf{s}_K(t; \boldsymbol{\theta}) 
\end{bmatrix} +  \begin{bmatrix}
\mathbf{n}_1(t)  \\
\vdots \\
\mathbf{n}_K(t)

\end{bmatrix}
$$
综上所述，EM算法对于多源信号叠加问题的求解步骤为（假设参数迭代到了$\boldsymbol{\theta}^{(n)}$）

- E-step：对于所有$k$，计算隐变量的观测数据$\hat{\mathbf{x}}_k(t) = \mathbf{s}_k(t; \boldsymbol{\theta}^{(n)}) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K} \mathbf{s}_l(t; \boldsymbol{\theta}^{(n)}) \right]$；
- M-step：在隐变量的模型上求解隐变量模型的最大似然估计$\boldsymbol{\theta}^{(n + 1)} = \arg \max\limits_{\boldsymbol{\theta}} \sum\limits_{k = 1}^{K} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}) \right]$。

EM算法是否能够简化运算的关键在于M-step的最大似然估计问题相较于观测变量的最大似然估计是否更加简单，从直觉来讲，多源信号叠加模型的隐变量设置将叠加在一起的时域信号拆开，由此隐变量的的估计问题是多源叠加信号拆开为单源信号后进行的，一般而言，单源信号的估计问题要比多源信号估计问题要更加简单。同时，M-step的运算并不要求一定要是最大似然估计，只要比原来更好就行（隐变量模型的次优解），这样也能保证EM算法具有单调不减的特性，即Generalized EM算法。

如果假设每个源的信号由单独且互不重叠的参数控制，那么M-step的多维参数估计问题可以拆开为并行的每个源单独的参数估计问题，M-step修改为

- M-step：对于所有$k$，在隐变量模型的单源信号上求解其的最大似然估计$\boldsymbol{\theta}^{(n + 1)}_k = \arg \max\limits_{\boldsymbol{\theta}_k} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}_k) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{s}_k(t; \boldsymbol{\theta}_k) \right]$。

相较于全局共享的参数，每个源的参数估计问题可以并行单独处理，计算复杂度降低。

**EM algorithm for DOA estimation**

现在考虑将EM算法运用在DOA估计问题上，其满足每个源都由独立的参数控制的条件
$$
\mathbf{s}_k(t; \boldsymbol{\xi}_k) = \mathbf{a}(\theta_k) s_k(t) \\
\boldsymbol{\xi}_k = \begin{bmatrix}
\theta_k & \bar{\mathbf{s}}_k^T & \tilde{\mathbf{s}}_k^T
\end{bmatrix}^T
$$
所以算法步骤为

- E-step：对于所有$k$，计算隐变量的观测数据$\hat{\mathbf{x}}_k(t) =  \mathbf{a}(\theta_k^{(n)}) s_k^{(n)}(t) + \beta_k \left[ \mathbf{y}(t) - \sum\limits_{l = 1}^{K}  \mathbf{a}(\theta_l^{(n)}) s_l^{(n)}(t) \right]$；
- M-step：对于所有$k$，在隐变量模型的单源信号上求解其的最大似然估计$\boldsymbol{\xi}^{(n + 1)}_k = \arg \max\limits_{\boldsymbol{\xi}_k} \sum\limits_{t = 1}^{N} \left[ \hat{\mathbf{x}}_k(t) -  \mathbf{a}(\theta_k) s_k(t) \right]^H \mathbf{Q}^{-1} \left[\hat{\mathbf{x}}_k(t) -  \mathbf{a}(\theta_k) s_k(t) \right]$。

可以在EM中使用分离线性参数的技巧（线性参数估计量可由非线性参数估计量表示），使得收敛速度更快。即在M-step中，我们不需要正真找到线性参数的最大似然估计量，只需要找到非线性参数的最大似然估计，而对于单源DOA估计的最大似然估计，就是周期功率谱的最大频率点。在求出所有非线性参数后，本轮得到的线性参数的最大似然估计就是非线性参数的变换（线性模型），由此可以重新改写M-step为

- M-step：对于所有$k$，在隐变量模型的单源信号上求解非线性参数的最大似然估计$\theta^{(n + 1)}_k = \arg \max\limits_{\theta_k} \sum\limits_{t = 1}^{N} \| \mathbf{a}^H(\theta_k) \mathbf{x}(t) \|^2$。在得到了所有源的DOA估计后，估计线性参数$\mathbf{s}^{(n)}(t) = (\mathbf{A}^H\mathbf{A})^{-1} \mathbf{A}^H \mathbf{y}(t)$。

在实践中，EM算法的鲁棒性很差，在角度离得近和信噪比低的情况下，算法要么有偏，要么发散，要么收敛到错误的解上。

### Gauss-Newton

DOA中的确定性最大似然估计DML从子空间的角度来说，可以视为一个subspace fitting的问题；从优化的角度来说，可以视为可分离变量的非线性最小二乘问题。数学表达如下
$$
\min _{\boldsymbol{\theta}, \mathbf{S}} \| \mathbf{X} - \mathbf{A} \mathbf{S} \|_F^2
$$
经过分离线性参数的技巧，可以得到该非线性最小二乘的优化函数为
$$
\begin{aligned}
f(\boldsymbol{\theta}) &= \frac{1}{2} \| \mathbf{P}_{A}^{\perp} \mathbf{X} \|_F^2 \\
&= \frac{1}{2} \| \mathbf{R}(\boldsymbol{\theta}) \|_F^2 \\
&= \frac{1}{2} \mathbf{r}^H \mathbf{r}
\end{aligned}
$$
其中，$\mathbf{r} = \operatorname{vec}\left[\mathbf{R}(\boldsymbol{\theta})\right]$。为了使用Gauss-Newton方法，我们需要求出$\mathbf{r}$的Jacobian矩阵，Jacobian矩阵的某一列为
$$
\begin{aligned}
\mathbf{J}({\theta_i}) &= \frac{\partial \mathbf{r}}{\partial {\theta}_i} \\
&= \frac{\partial \operatorname{vec}\left[\mathbf{R}(\boldsymbol{\theta})\right]}{\partial {\theta}_i} \\
&= \operatorname{vec} \left[ \frac{\partial \mathbf{R}(\boldsymbol{\theta})}{\partial \theta_i} \right]
\end{aligned}
$$

$$
\mathbf{J}(\boldsymbol{\theta}) = \begin{bmatrix}
\operatorname{vec} \left[ \frac{\partial \mathbf{R}(\boldsymbol{\theta})}{\partial \theta_1} \right] & \cdots & \operatorname{vec} \left[ \frac{\partial \mathbf{R}(\boldsymbol{\theta})}{\partial \theta_K} \right]

\end{bmatrix}
$$

由此，在具体计算Jacobian矩阵上，我们可以先求出矩阵$D \mathbf{R}(\boldsymbol{\theta})$，然后再将其矩阵的维度拉直，从而降为成一个新的矩阵，就是Jacobian矩阵。

接着按照正常的Gauss-Newton的流程走完就行，唯一需要注意的地方是backtracking在离最优点很近的时候，可能会陷入死循环，这是算法具体实现的问题。其他更加细节的推导看Separable Nonlinear Least Squares。













