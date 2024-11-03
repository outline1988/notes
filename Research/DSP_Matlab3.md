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
其中，$A = \frac{1}{2}C_{uu}$且$B = \frac{1}{2}C_{vu}$。换句话说，复随机变量和实随机变量只是同一个东西不同的表现形式，使用复数的计算会更加简单，但是虽然是复数，但是本质上还是满足特定关系的实随即矢量的PDF。使用复随机变量只是为了形式上的方便。

现在证明，在$C_{uu} = C_{vv}$以及$C_{uv} = -C_{vu}$的条件下$p(\mathbf{\tilde{x}}) = p(\mathbf{x})$。该结论等效证明两个性质。P381

### filter

它支持**有限冲激响应 (FIR)** 和**无限冲激响应 (IIR)** 滤波器。

```matlab
y = filter(b, a, x)
```

假设滤波器的传递函数为：
$$
H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_mz^{-m}}{1 + a_1 z^{-1} + \cdots + a_n z^{-n}}
$$
所以`b`是一个$m + 1$长的向量，第一个元素$b_0$不一定为1，但是其代表整体的一个线性系数。`a`是一个$n + 1$的向量，其第一个元素$a_0$必须为1。

*关于暂态响应的问题没有讨论，以后讨论。*

### periodogram

如果$N = 512$，如果按照正常的使用FFT去做周期图，则需要将$\lfloor N / 2 \rfloor = 512$的点归为负频率，因为`fftshift`函数就是这样做的，但是`peridogram`函数将$\lfloor N / 2 \rfloor + 1 = 513$个点归为正频率，也即他认为$f = 256 / 512$这个频率属于正频率，也没错，只不过平时使用`fftshift`将这个点归为负频率。



### 元胞数组

元胞数组中的打包和解包操作。

```matlab
% 定义输入矩阵 A 和对角块数量 N
A = [1, 2; 3, 4];  % 一个 2x2 的矩阵
N = 3;  % 对角块的数量

% 创建块对角矩阵
c = repmat({A}, 1, N);
B = blkdiag(c{:});

% 输出结果
disp(B);
```

比如需要一个保护N个矩阵A的分块对角矩阵，然而blkdiag函数只能够接收给定数量的矩阵来形成分块对角，而我们需要自定义K个矩阵，那么可以使用cell的操作来产生K个输入自变量。

首先可以使用repmat函数来创建重复的cell类型。

然后在函数的输入参数中使用{:}进行类似解包的操作。
