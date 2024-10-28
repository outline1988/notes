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
其中矩阵$D$为可逆方阵，其可以作为一个随机矢量的白化滤波器。、

