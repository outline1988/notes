### Separable Nonlinear Least Squares (SNLLS)

参考：On Some Separated Algorithms for Separable Nonlinear Least Squares Problems

The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate

本节不会考虑太多关于优化方法的讨论，而是如何将优化对应的优化方法应用到SNLLS问题中，关于优化的部分可以参考Numerical Optimization. 

考虑这样一个非线性最小二乘问题
$$
\min_{\mathbf{x}, \mathbf{c}} f(\mathbf{x}, \mathbf{c}) = \min_{\mathbf{x}, \mathbf{c}} \frac{1}{2} \left\| \mathbf{y} - \mathbf{A}(\mathbf{x}) \mathbf{c} \right\| ^2
$$
其中，$\mathbf{x}$为非线性部分的参数，$\mathbf{c}$为线性部分的参数。可以使用参数分离的技巧，将线性参数$\mathbf{c}$用非线性参数$\mathbf{x}$来表示
$$
f(\mathbf{x}) =  \frac{1}{2} \left\| \mathbf{P}_{A}^{\perp}\mathbf{y}\right\| ^2 = \frac{1}{2} \mathbf{r}^H(\mathbf{x}) \mathbf{r}(\mathbf{x})
$$
如此便只需要对非线性参数进行优化。

考虑使用Newton方法来进行line search，则需要求出其梯度和Hessian矩阵
$$
\nabla f(\mathbf{x}) = \operatorname{Re}(\mathbf{J}^H \mathbf{r}) \\
$$

$$
\nabla^2 f(\mathbf{x}) = \operatorname{Re} \left(\mathbf{J}^H \mathbf{J} + \sum\limits_{i = 1}^{m} r_i \nabla^2 r_i \right)
$$

其中，$\mathbf{J}$为Jacobian矩阵
$$
\mathbf{J}(\mathbf{x}) = \frac{\partial \mathbf{r}(\mathbf{x})}{ \partial \mathbf{x}^T} = \begin{bmatrix}
\frac{\partial r_1(\mathbf{x})}{\partial \mathbf{x}^T} \\
\vdots \\
\frac{\partial r_m(\mathbf{x})}{\partial \mathbf{x}^T}
\end{bmatrix} = \begin{bmatrix}
\nabla r_1^T \\
\vdots \\
\nabla r_m^T
\end{bmatrix}
$$
Gauss-Newton方法将hessian矩阵的第二项该省略，由此只需要知道关于$\mathbf{r}(\mathbf{x})$的Jacobian矩阵就可以完成后面的line search。

接下来考虑如何求出Jacobian矩阵
$$
\begin{aligned}
\mathbf{J}(\mathbf{x}) &= \frac{\partial \mathbf{r}(\mathbf{x})}{ \partial \mathbf{x}^T} \\
&= \frac{\partial }{ \partial \mathbf{x}^T} \mathbf{P}_{A}^{\perp}\mathbf{y} \\
&= \frac{\partial \mathbf{P}_{A}^{\perp}}{ \partial \mathbf{x}^T} \mathbf{y}
\end{aligned}
$$
这里涉及到了向量对一个矩阵求导，所以引入新的符号$D \mathbf{A}$，可以将其理解为在原先二维矩阵的基础上增加了一个维度pages，每一页就是原矩阵对于某一个标量变量的求导，最终是一个tridimensional slabs。现在我们只考虑其与二维以下的变量进行运算，其与任何二维矩阵和一维向量进行运算，都先不考虑增加的第三维度，先按照原先两维的方式进行运算，再在最后的结果上增加第三个维度。比如，$D \mathbf{A} \mathbf{B}$就是先计算$\frac{\partial  \mathbf{A}}{\partial x_i} \mathbf{B}$，然后在把所有矩阵堆叠形成一个tridimensional slabs。以上操作对应于matlab函数为`pagemtimes`函数。

对于投影矩阵的求导，需要用到以下这几个性质

- 若$\mathbf{B}$的列空间位于$\mathbf{A}$列空间内，则
  $$
  D \mathbf{P}_A \mathbf{B} = \mathbf{P}_{A}^{\perp} D \mathbf{B}
  $$

- 若$\mathbf{A}$和$\mathbf{B}$都是Hermitian矩阵
  $$
  (D \mathbf{A}\mathbf{B})^H = \mathbf{B} D \mathbf{A}
  $$

由此
$$
\begin{aligned}
D \mathbf{P}_A = D \mathbf{P}_A^2 = D \mathbf{P}_A \mathbf{P}_A  + \mathbf{P}_A D \mathbf{P}_A 
\end{aligned}
$$

$$
D \mathbf{P}_A \mathbf{P}_A = D  \mathbf{P}_A \mathbf{A} \mathbf{A}^{\dagger} = \mathbf{P}_A^{\perp} D \mathbf{A} \mathbf{A}^{\dagger}
$$

$$
\mathbf{P}_A D \mathbf{P}_A  = (D \mathbf{P}_A  \mathbf{P}_A )^H
$$

故最终
$$
D \mathbf{P}_A = \mathbf{P}_A^{\perp} D \mathbf{A} \mathbf{A}^{\dagger} + (\mathbf{P}_A^{\perp} D \mathbf{A} \mathbf{A}^{\dagger})^H
$$
Kaufman给出了投影矩阵求导的更简单的版本，即去掉第二项
$$
D \mathbf{P}_A \approx \mathbf{P}_A^{\perp} D \mathbf{A} \mathbf{A}^{\dagger}
$$
由此，便完成了Jacobian矩阵的计算，Gauss-Newton可以随之进行。

### 矩阵范数相关

若有两个大小相同的矩阵，希望计算其对应元素相乘后求和的形式。首先，最简单的，将矩阵拉直后内积
$$
\operatorname{vec}(B)^H \operatorname{vec}(A)
$$
其次，就是使用Frobenius内积的定义，见下一节。

再次，就是使用迹的性质，这是最不好理解，也最容易忘的。回想以下两个矩阵相乘是左边矩阵的行与右边矩阵的列对应相乘后求和，那么对角线上的元素就是不同行与列对应相乘后求和，所以使用迹加在一起。

#### Frobenius系列

对于两个维数相同的复数矩阵，其Frobenius内积定义为
$$
\langle A, B \rangle_F = \sum\limits_{i, j}a_{ij}b_{ij}^*
$$
其有一个便于计算的性质，就是岂能用迹表示出来
$$
\langle A, B \rangle_F = \operatorname{tr}(A B^H)
$$
由此，Frobenius范数定义为
$$
\| A \|_F^2 = \langle A, A \rangle = \sum\limits_{i, j}a_{ij}a_{ij}^* = \operatorname{tr}(A A^H)
$$

$$
\mathbf{y}(t) = \begin{bmatrix}
y_{1}(t)  \\
y_{2}(t) \\
\vdots \\
y_{M}(t) \\
\end{bmatrix} = \begin{bmatrix}
1 \\
e^{\mathrm{j}2\phi} \\
\vdots \\
e^{\mathrm{j}(M-1)\phi}
\end{bmatrix} s(t)
$$