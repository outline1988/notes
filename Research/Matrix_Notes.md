## Linear Algebra

### 线性变换视角下的矩阵$A$

对于任意矩阵$A \in \mathbb{R}^{m \times n}$，以变换的角度来看待，其都为一个其**行空间到列空间的线性变换**，其中**行空间**为矩阵$A$所有行向量所张成的$n$维空间下的$r$维子空间，同理，**列空间**为矩阵$A$所有列向量所张成的$m$维空间下的$r$维子空间。注意转换前后的空间所处的总空间不同（$n$维和$m$维空间），但是子空间维度是相同的，并且行子空间与列子空间是一一对应的。举例来说，对于一个$A \in \mathbb{R}^{3 \times 2}$的矩阵，其表示的线性变换为一个整个2维的行空间到一个3维空间中的一个$2$维子空间（平面）的一一映射。

![image-20240513202550561](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240513202550561.png)

对于$A \boldsymbol{x} = \boldsymbol{b}$来说，任意$\boldsymbol{x}$张成整个$n$维空间，而此时$\boldsymbol{b}$只张成$m$维空间的$r$维列空间。对于$\boldsymbol{x}$的整个$n$维空间来说，仅有一小部分的$r$维行空间与列空间是一一对应的，与之正交的剩余子空间为零空间，即任意$\boldsymbol{x} = \boldsymbol{x}_r + \boldsymbol{x}_n$，且$\boldsymbol{x}_r \perp \boldsymbol{x}_n$，满足
$$
\begin{aligned}
A \boldsymbol{x} 
&= A (\boldsymbol{x}_r + \boldsymbol{x}_n) \\
&= A\boldsymbol{x}_r + A \boldsymbol{x}_n \\
&= \boldsymbol{b} + \boldsymbol{0} \\
&= \boldsymbol{b}
\end{aligned}
$$

同样，可以画根据伪逆的公式来画一个类似的图，以证明伪逆与原矩阵的某种对称性
$$
A^{\dagger} \boldsymbol{b} = \boldsymbol{x}_r
$$


### 特征值和特征向量

#### 简介

当矩阵$A \in \mathbb{R}^{n}$方阵，那么变换前后的行空间和列空间同属于一个大的$n$维度空间，当$\text{rank}(A) = n$时，$A$的线性变换就表示同一空间中不同向量与不同向量的转换，例如旋转矩阵；当$\text{rank}(A) \leq n$时，表示同一空间中的两个子空间（可能相同也可能不同）的不同向量的转换。

当使用方阵$A\in \mathbb{R}^{n}$进行$n$维空间到$n$维空间的转换时，除了一一对应的行空间和列空间值得关注，还有另外一些子空间值得注意，称为特征空间（有多个），每个特征空间对应一个特征值，处于特征空间的特征向量不会由于$A$的作用下而发生方向的变化，只可能发生幅度的变换，也即
$$
A \boldsymbol{x} = \lambda \boldsymbol{x}
$$
最显而易见的特征空间就是$A$的零空间（与行空间正交），对应于特征值$\lambda = 0$。

特征空间最小的维度是1，也即特征空间最少是一条直线，特征空间的维度可能超过1，也即一个平面或是一个超平面。有一个直观的感受就是如果线性变换$A$的线性操作对与行空间每一个维度都单独进行变换，那么特征空间一般都是一维，如果对行空间多个维度同时进行变换，那么特征空间一般为与之对应的多维（只是直观的感受，可能不准确）。

特征值会随着行变换的高斯消元而发生变换，原始矩阵的特征值与rref的矩阵特征值不同，后者的特征值是pivots。但是特征值之积就是矩阵的行列式，这一点可以通过特征值分解来得出；特征值之和为矩阵的迹（trace）。

#### 矩阵对角化

矩阵的所有特征值都不同，那么所有的特征向量都是线性独立的，所以必能对角化。

证：假设$\boldsymbol{x}_1$和$\boldsymbol{x}_2$是两个特征值$\lambda_1$和$\lambda_2$对应的特征向量。反证法，假设两个特征向量线性相关，则
$$
c_1 \boldsymbol{x}_1 + c_2 \boldsymbol{x}_2 = 0
$$
两边同时乘$A$，两边依然成立$c_1 \lambda_1 \boldsymbol{x}_1 + c_2 \lambda_2 \boldsymbol{x}_2 = 0$。两边同时乘$\lambda_2$，两边也依然成立$c_1 \lambda_2 \boldsymbol{x}_1 + c_2 \lambda_2 \boldsymbol{x}_2 = 0$。两式相减，则有
$$
(\lambda_1 - \lambda_2) c_1 \boldsymbol{x}_1 = 0
$$
由于$\lambda_1 \neq \lambda_2$，则$c_1 = 0$，与假设矛盾，所以必有不同特征值对应的特征向量一定线性独立。

可对角化和可逆没有明显的联系，前者关注特征向量之间是否线性独立，后者关注特征值有没有为0。

#### 对称矩阵

前面所针对的矩阵的特征值分解是任意的，现在我们开始针对一类特殊的矩阵$S$，就是我们熟知的对称矩阵$S^{\top} = S$。从行空间线性变换为列空间的角度来说，行空间和列空间为同一子空间，同时张成的基向量也相同。

**Spectral Theorem**
对称矩阵$S$的特征值分解可选择为如下
$$
S = Q \Lambda Q^{-1} = Q \Lambda Q^{\top}
$$
其中$Q$为正交矩阵，所以满足$Q^{\top} = Q^{-1}$。

对称矩阵不同特征值对应的特征空间必定正交，证明：取点积
$$
\lambda_1 \boldsymbol{x}^{\top} \boldsymbol{y} = (\lambda_1 \boldsymbol{x})^{\top} \boldsymbol{y} = \boldsymbol{x}^{\top} S \boldsymbol{y} = \lambda_2 \boldsymbol{x}^{\top} \boldsymbol{y}
$$
由于$\lambda_1 \neq \lambda_2$，故$\boldsymbol{x}^{\top} \boldsymbol{y} = 0$正交。同时对称矩阵保证可以对角化（P342）。

举个例子，检测理论中广义匹配滤波器优化的目标为
$$
d^2 = \mathrm{s}^{\top} C^{-1} \mathrm{s}
$$
在能量约束的情况下$\mathcal{E} = \mathrm{s}^{\top}\mathrm{s}$希望能找到最合适的$\mathrm{s}$使得目标函数$d^2$最大。将正定矩阵$C^{-1}$进行特征值分解（谱分解），则
$$
\begin{aligned}
d^2 &= \mathrm{s}^{\top} Q \Lambda^{-1} Q^{\top}  \mathrm{s} \\
&= \| \Lambda^{-1 / 2} Q^{\top} \mathrm{s} \| ^2
\end{aligned}
$$
其中$Q^{\top}$对$\mathrm{s}$以正交的特征向量为基进行分解，在对每个系数乘上相应的系数$\Lambda^{-1 / 2}$，系数是$C$的固有属性，即特征值。优化的结果为$\mathrm{s}$应该设计为$C$的最小特征值的特征向量，由此用特征向量对$\mathrm{s}$进行分解后，只有最小特征向量上有系数，这个系数会被$\Lambda^{-1 / 2}$得到最大的放大。

**特征值和特征向量的对应关系**
$$
\text{product of pivots} = \text{product of eigenvalues} = \text{determinant}
$$
对于**对称矩阵**来说，正特征值的数量和正特征向量的数量相同，这对于正定矩阵的判断具有重要作用。

#### 正定矩阵

对称矩阵的基础上满足以下五种判断正定矩阵的方式，则为正定矩阵

- 所有eigenvalues大于0（定义）；
- 所有pivots大于0（对称矩阵的特征值和pivots符号相对应）；
- 左上角的行列式都大于0（pivots都大于0）;
- $\boldsymbol{x}^{\top} S \boldsymbol{x} > 0$对于任意$\boldsymbol{x}$都成立；
- 可分解为$S = A^{\top} A$的形式（充要条件），其中$A$必须有线性独立的列向量。

对于最后一条，正定矩阵可以用多种方式进行分解（P352）。

正定矩阵必定可以Cholesky分解，即
$$
C = D^{\top}D
$$
其中矩阵$D$为可逆方阵，其可以作为一个随机矢量的白化滤波器。

#### 条件数与线性方程的敏感度

#### 广义逆



### 向量链式法则

参考：The Matrix Calculus You Need For Deep Learning

经过微积分的学习后，标量对标量的求导应该了然于心。但是随着向量的不断出现，我们需要克服标量对向量的求导，向量对向量的求导这两种类型。同时我们希望微积分已经学过的链式求导法则依然适用在向量的求导中，首先需要了解三个基本法则：

- **单变量单链的求导法则**：顾名思义，在整个变量的传递过程中，一直是以单变量单链进行传递的，比如$y = \sin(x^2)$的传递过程为$x \rightarrow x^2 \rightarrow \sin(x^2)$，这种单变量单链的求导是最简单的，可以直接套用链式法则。

- **单变量多链的求导法则**：不同于单链的求导法则，最底层的单变量$x$可以以多种路径（多链）的方式来最终影响函数值，比如$y = \sin(x + x^2)$，变量$x$一条路径为$x \rightarrow x^2 \rightarrow \sin(x^2 + u_2)$，另一条路径为$x \rightarrow \sin(u_1 + x)$，对于这样的多链求导，求导法则可以总结为
  $$
  \frac{\partial f(u_1,\ldots,u_{n+1})}{\partial x}=\sum_{i=1}^{n+1}\frac{\partial f}{\partial u_i}\frac{\partial u_i}{\partial x}
  $$
  多链的每一条链的求导都按照单链求导法则进行，最后再所有链加在一起。

**向量单链的求导法则**单独列出：对于向量的求导法则，习惯上主要包括两种，一是标量对列向量求导，求导的结果是列向量；二是列向量对行向量求导，求导的结果是一个矩阵（雅可比矩阵的形式）。然而，这两种方式在求导公式的统一上是不和谐的，因为向量的求导有两种准则，一是分子布局，二是分母布局（具体不同的布局是怎样的不需要关心），只需要知道习惯的标量对向量求导和向量对向量求导的排列方式是不同的布局，这就给了统一的公式造成了困难。

所以为了统一，在向量对向量求导的情形下，规定（雅可比矩阵形式）
$$
\left[\frac{\mathrm{d} \mathbf{f}(\mathbf{x})}{\mathrm{d}\mathbf{x}^T}\right]_{ij} = \frac{\mathrm{d} f_i(\mathbf{x})}{\mathrm{d} x_j}
$$
对于雅可比形式的链式求导，可直接使用公式
$$
\frac{\mathrm{d} \mathbf{f}\left( \mathbf{g}(\mathbf{x}) \right)}{\mathrm{d} \mathbf{x}^T} = \frac{\partial\mathbf{f}}{\partial\mathbf{g}^T}\frac{\partial\mathbf{g}}{\partial \mathbf{x}^T}
$$
而标量对向量的求导可以视为雅可比矩阵形式的转置
$$
\begin{aligned}
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \left[\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}^T}\right]^T 
\end{aligned}
$$
所以求导可以借助雅可比形式的链式法则
$$
\begin{aligned}
\frac{\mathrm{d} f(\mathbf{g} \left( \mathbf{x} \right) )}{\mathrm{d} \mathbf{x}} &= \left[\frac{\mathrm{d} f(\mathbf{g} \left( \mathbf{x} \right) )}{\mathrm{d} \mathbf{x}^T}\right]^T \\
&= \left[\frac{\partial{f}}{\partial\mathbf{g}^T}\frac{\partial\mathbf{g}}{\partial \mathbf{x}^T}\right]^T \\
\end{aligned}
$$
几个例子：
$$
\begin{aligned}
\frac{\mathrm{d} (\mathbf{x}^T \mathbf{A} \mathbf{x})}{\mathrm{d} \mathbf{x} } &= 
\left[\frac{\mathrm{d} (\mathbf{x}^T \mathbf{A} \mathbf{x})}{\mathrm{d} \mathbf{x} ^T}\right]^T \\
&= \left[\frac{\mathrm{d} (\mathbf{u}_1^T \mathbf{u}_2)}{\mathrm{d} \mathbf{x}^T} \right]^T\\
&= \left[\frac{\mathrm{d} (\mathbf{u}_1^T \mathbf{u}_2)}{\mathrm{d} \mathbf{u}_1^T}  \frac{\mathrm{d} \mathbf{u}_1}{\mathrm{d} \mathbf{x}^T} + \frac{\mathrm{d} (\mathbf{u}_1^T \mathbf{u}_2)}{\mathrm{d} \mathbf{u}_2^T}  \frac{\mathrm{d} \mathbf{u}_2}{\mathrm{d} \mathbf{x}^T} \right]^T \\
&= \mathbf{A}\mathbf{x} + \mathbf{A}^T\mathbf{x}
\end{aligned}
$$
### Total Least Square

考虑一个二维曲线拟合的问题，给定坐标点$\left\{(x_i, y_i)\right\}_{i = 1}^{N}$，我们使用线性模型来拟合这条曲线
$$
\mathbf{y} \approx \mathbf{X} \mathbf{a}
$$
其中，$\mathbf{y}$就是输出数据$\left\{y_i\right\}_{i = 1}^{N}$组成的列向量，而$\mathbf{X} \in \mathbb{R}^{m \times n}$是由输入数据$\left\{x_i\right\}_{i = 1}^{N}$经过预处理（比如多项式函数）而组成的矩阵，满足$m > n$，是列满秩矩阵。例如需要拟合的模型为
$$
y \approx a_1 x + a_2 x^2
$$
则我们可以写出线性模型为
$$
\begin{bmatrix}
y_1 \\
\vdots \\
y_N
\end{bmatrix} = \begin{bmatrix}
x_1 & x_1^2 \\
\vdots & \vdots \\
x_N & x_N^2
\end{bmatrix} 
\begin{bmatrix}
a_1 \\
a_2
\end{bmatrix}
$$
在给定了所有的坐标点后，需要我们确定的就是参数向量$\mathbf{a}$。显然，在学习了LS方法后，可以直接写出参数的估计量为
$$
\hat{\mathbf{a}}_{\text{LS}} = \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T \mathbf{y}
$$
物理意义就是将向量$\mathbf{y}$投影到$\mathbf{X}$的列空间中。同时从另一个角度来看，LS找到了最小二范数$\| \Delta \mathbf{y} \|_2$，使得下式成立
$$
\mathbf{y} + \Delta\mathbf{y} = \mathbf{X} \mathbf{a}
$$
注意，这个$\Delta \mathbf{y}$与$\mathbf{X}$的列空间是正交的，所以才会使二范数最小。

然而，现实情况中，输入数据$\mathbf{x}$和输出数据$\mathbf{y}$都有可能受到扰动，而最小二乘默认假设输入数据$\mathbf{x}$是精确的，一个更自然的想法是使得下式矩阵的Frobenius范数在某个约束下最小（至于为什么不是其他范数，这是为了求解的方便）
$$
\min \left\| \begin{bmatrix}
\Delta\mathbf{X} & \Delta\mathbf{y}
\end{bmatrix} \right\|^2_\text{F} \\
\text{s.t.} \quad \mathbf{y} + \Delta\mathbf{y} = (\mathbf{X} + \Delta\mathbf{X}) \mathbf{a}
$$
Frobenius范数是矩阵范数的一种，满足以下特性
$$
\left\| \mathbf{Z} \right\|^2_\text{F} = \sum_{i, j} z_{ij}^2 = \operatorname{tr}(\mathbf{Z}^T\mathbf{Z}) = \sum_i \sigma^2_i
$$
其中，$\sigma_i$是矩阵的奇异值。

从直观意义上来说，我们假设输入数据和输出数据都受到的扰动，故我们需要找到最小的扰动项，使得线性模型的关系成立，此时隐含着假设了输入数据的扰动$\Delta \mathbf{X}$和输出数据的扰动$\Delta \mathbf{y}$的数量级是相当的。

Eckart-Young定理：给定一个秩为$n$的矩阵$\mathbf{Z}$，使得$\left\| \hat{\mathbf{Z}} - \mathbf{Z} \right\|^2_{\text{F}}$最小的秩$\hat{n}$矩阵（$\hat{n} < n$）为矩阵$\mathbf{Z}$进行奇异值展开后舍弃最小的$n - \hat{n}$个奇异值而形成的矩阵$\hat{\mathbf{Z}}$。

我们可以将上述优化问题的约束方程换一种形式
$$
\min \left\| \begin{bmatrix}
\Delta\mathbf{X} & \Delta\mathbf{y}
\end{bmatrix} \right\|^2_\text{F} \\
\text{s.t.} \quad \begin{bmatrix}
\mathbf{X} + \Delta\mathbf{X} &  \mathbf{y} + \Delta\mathbf{y}
\end{bmatrix} \begin{bmatrix}
-\mathbf{a} \\
1
\end{bmatrix}  = \mathbf{0}
$$
从上式可以看出，向量$\begin{bmatrix} \mathbf{a} &1 \end{bmatrix}^T$为矩阵$\begin{bmatrix} \mathbf{X} + \Delta\mathbf{X} &  \mathbf{y} + \Delta\mathbf{y} \end{bmatrix} \in \mathcal{R}^{m \times (n + 1)}$的零空间，由于假设$\mathbf{X}$为列满秩矩阵，所以该矩阵的秩一定为$n$。

同时，我们得到数据形成的矩阵$\begin{bmatrix} \mathbf{X} &  \mathbf{y} \end{bmatrix} \in \mathcal{R}^{m \times (n + 1)}$由于受到的扰动，其秩为$n + 1$，所以受到了Eckart-Young定理的影响，我们就可以通过将矩阵$\begin{bmatrix} \mathbf{X} &  \mathbf{y} \end{bmatrix}$奇异值展开并抛弃最小值来得到一个秩为$n$的矩阵，而这个矩阵正是我们想要找到的满足上述优化问题的矩阵$\begin{bmatrix} \mathbf{X} + \Delta\mathbf{X} &  \mathbf{y} + \Delta\mathbf{y} \end{bmatrix}$，然后就能够通过找到该矩阵的零空间的方式来得到TLS解。

可以更好的简化上述的流程，对矩阵进行奇异值展开
$$
\begin{aligned}
\begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} &= \sum\limits_{i = 1}^{n + 1} \sigma_i \mathbf{u}_i \mathbf{v}_i^T \\
&= \sum\limits_{i = 1}^{n} \sigma_i \mathbf{u}_i \mathbf{v}_i^T +  \sigma_{n + 1} \mathbf{u}_{n + 1} \mathbf{v}_{n + 1}^T  \\
&= \begin{bmatrix} \mathbf{X} + \Delta\mathbf{X} &  \mathbf{y} + \Delta\mathbf{y} \end{bmatrix} + 
\begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix}  \mathbf{v}_{n + 1} \mathbf{v}_{n + 1}^T
\end{aligned}
$$
两边同时右乘最优一个右奇异向量$\mathbf{v}_{n + 1}$，并交换位置
$$
\begin{aligned}
\begin{bmatrix} \mathbf{X} + \Delta\mathbf{X} &  \mathbf{y} + \Delta\mathbf{y} \end{bmatrix} \mathbf{v}_{n + 1} &= \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} \mathbf{v}_{n + 1} - \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix}   \mathbf{v}_{n + 1} \mathbf{v}_{n + 1}^T \mathbf{v}_{n + 1} \\
&= \mathbf{0}
\end{aligned}
$$
所以我们可以轻松得到解
$$
\hat{\mathbf{a}}_{\text{TLS}} = -\frac{ \left [ \mathbf{v}_{n + 1}\right]_{1 : n}} {\left [ \mathbf{v}_{n + 1}\right]_{n + 1}}
$$
注意到
$$
\begin{aligned}
\begin{bmatrix}
\Delta\mathbf{X} & \Delta\mathbf{y}
\end{bmatrix} &= \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix}  \mathbf{v}_{n + 1} \mathbf{v}_{n + 1}^T \\
&= \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} \mathbf{P}_{v}
\end{aligned}
$$
从这个角度来说，TLS的残差项就是增广矩阵的行向量投影到行奇异空间中对应奇异值最小的子空间中。既然能投影到行奇异空间的子空间中，那么为什么不可以投影到列奇异空间的子空间中呢？所以做以下尝试
$$
\begin{aligned}
\mathbf{P}_{u} \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} &=  
\mathbf{u}_{n + 1} \mathbf{u}_{n + 1}^T \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} \\
&= \mathbf{u}_{n + 1} \mathbf{u}_{n + 1}^T \sum\limits_{i = 1}^{n + 1} \sigma_i \mathbf{u}_i \mathbf{v}_i^T \\
&= \sigma_{n + 1} \mathbf{u}_{n + 1} \mathbf{v}_{n + 1}^T \\
&= \begin{bmatrix} 
\mathbf{X} &  \mathbf{y} 
\end{bmatrix} \mathbf{P}_{v}
\end{aligned}
$$
可以看到，从投影到奇异空间的角度来说，行向量投影和列向量投影是等价地，这也符合直觉，因为奇异值分解的本质就是找到了行空间和列空间的正交基。

由此可以进行总结，TLS的方法本质上同LS一样都是进行降维操作，前者在于将增广矩阵$\begin{bmatrix} \mathbf{X} &  \mathbf{y} \end{bmatrix}$进行降维，而后者在于将$\mathbf{y}$降维至与$\mathbf{X}$同一列空间中。TLS更加贴近传统意义上的PCA算法，其依靠的正是SVD分解寻找行空间与列空间正交基的能力。

