### 功率谱估计

对于一般的确知信号来说，其时域通常是能量有限的，即时域满足Dirichlet条件。其中有一例外就是周期信号，虽然其不满足Dirichlet条件，但是将其用广义函数（冲激函数等）推广后，就可以得到为离散冲激的频谱函数。

对于平稳的随机信号来说，由于其平稳的特性，其通常不满足Dirichlet条件，故不存在傅里叶变换，但其为功率有限信号，故存在功率谱密度函数。Wiener-Khinchin定理指出，虽然其无法用傅里叶变换来表示其频谱特征，却可以使用功率谱密度来描述，同时该定理指出，功率密度函数正好是时域信号自相关函数的傅里叶变换。

功率谱密度以频率为横轴，$W/Hz$为纵轴，其描述的是随机信号的功率随不同频率变化的情况，即将随机信号的有限功率根据不同的频率分成不同的功率成分，并对应展现在频域当中，所以对功率谱密度函数积分的结果为随机信号的功率。

现在没时间写，以后再说，参考连接如下。

[经典功率谱估计及Matlab仿真]: https://www.cnblogs.com/jacklu/p/5140913.html

谱估计问题是一门很大的问题，涉及到的理论非常多，应该作为单独一门课程来学习。

### 一元随机变量的函数

**一元随机变量的函数的概率密度函数本质上是满足归一化条件下的复合函数的变换。**

对于一个随机变量$X$，可将其进行某个映射$g(\cdot)$，使其形成一个新的随机变量$Y = g(X)$，则此时新的随机变量$Y$的分布与$X$的分布和映射$g(\cdot)$息息相关。

随机变量的本质就是样本空间到数字域的确定性映射，$X$是一个从样本空间到数字域的确定性映射，而$g(\cdot)$是一个数字域到数字域的确定性映射，所以$Y = g(X)$也必然为样本空间到数字域的确定性映射，理所当然的也为一个 随机变量。

![image-20230517152045123](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230517152045123.png)

![image-20230517152059880](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230517152059880.png)

![image-20230517152116629](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230517152116629.png)

上述证明的本质在于使用了微小量$dx$和$dy$来划分区间，由此来联系随机随机变量$X$和$Y$。即（对于严格单调递增函数的情况）
$$
P\{x < X < x + \mathrm{d}x\} = P\{y < Y < y + \mathrm{d}y\}
$$
由于
$$
P\{x < X < x + \mathrm{d}x\} \approx f_X(x)\mathrm{d}x
$$
和
$$
P\{y < Y < y + \mathrm{d}y\} \approx f_Y(y)\mathrm{d}y
$$
故有
$$
f_X(x)\mathrm{d}x = f_Y(y)\mathrm{d}y
$$
由此便有
$$
f_Y(y) = f_X(x)\frac{\mathrm{d}x}{\mathrm{d}y}
$$
**连续变量的直方图均衡**
直方图均衡的本质在于将图像的归一化灰度图函数视为概率密度，此时每一级灰度（这里另其为连续的）视为随机变量，为了实现直方图均衡，就是找到一个随机变量的函数$s = T(r)$使得，新图像的被视为随机变量的灰度级别$s$的归一化灰度图函数（概率密度函数）为均匀分布的。

由前可知
$$
f_X(x)\mathrm{d}x = f_Y(y)\mathrm{d}y
$$
我们另新的随机变量函数$f_Y(y)$为均匀分布，那么就有
$$
f_Y(y) = \frac{1}{L - 1}, \ \ \ \ 0 \le y \le L - 1
$$
其中灰度级别的范围为$[0, L - 1]$。

那么便有
$$
\frac{\mathrm{d}y}{\mathrm{d}x} = \frac{f_X(x)}{f_Y(y)} = (L - 1)f_X(x)
$$
故可求得变换函数$y = g(x)$为
$$
g(x) = (L - 1) \int_0^x f_X(x)\mathrm{d}x
$$
将上述式子符号都转化为图像相关的符号，则有
$$
s = T(r) = (L - 1)\int_0^r p_r(w)\mathrm{d}w
$$
这便是直方图均衡的连续表达式推导。

**能量加权平均，校正fft的频谱泄露和栅栏效应问题**


> 第5章一元随机变量的函数. 概率、随机变量与随机过程

### 正交信号

不同于我们常遇见的随着时间的变化，实幅值发生变化的实信号，正交信号为随着时间的变化，复幅值发生变化的复信号。一言以蔽之，数学表达式只有实数的信号被称为实信号，数学表达式有复数的信号称为正交信号（复信号）。从正交信号的层面来看待，实信号属于正交信号的一种特殊形式。之所以被称为正交信号，是因为我们通常称正交信号复幅值的实部为同相部分，而虚部称为正交部分。总之，**正交信号就是复信号**。

复数的概念及其各种表达式，包括欧拉公式我们都很熟悉，故不再解释。在这里，我们将自然界存在的信号从实信号转变为了复信号，即我们认为实信号只是复数信号的投影。那么我们将如何传输这复信号呢？自然界只能通过实信号进行传输，而复信号可以轻易的根据欧拉公式拆解为实部和虚部之和的形式，故我们可以分别通过两个传输通道对复信号的实部和虚部同时进行传输，最后通过特定的方式将两个实信号还原为正交信号，该特定的方式必将包含对$j$算子的操作（如接收的$\cos(2\pi f_0 t)$接入示波器的X-Y模式的横轴，接收的$\sin(2 \pi f_0 t)$接入纵轴，由此便可在示波器中还原复指数函数$e^{j 2 \pi f_0 t}$，接入横轴和纵轴的方式已经暗含了我们对于$j$算子的处理）。

或者更简单来讲，我们认为世界上存在的任何信号都属于正交信号，研究都是正交信号，因为此时信号的数学形式更加简单。然而，自然界的限制使得我们只能传播实信号，为了能将正交信号与实信号一一对应，我们规定只研究正频率的正交信号，由此我们便能将为了数学方便研究的正频率正交信号一一对应为能在自然界传播的实信号。而完成正交信号与实信号相互唯一转换的方式就是欧拉公式与希尔伯特变换。

现在我们来考虑如何用正交信号来表示自然界中存在的实信号。考虑复指数信号$e^{j 2 \pi f_0 t}$和$e^{- j 2 \pi f_0 t}$，想象其在复数域的运动状态，可以很轻松知道，两个复向量以正实轴为起点，分别按照逆时针和顺时针的方向同角速度进行旋转。也就是说，将这两个信号相加，总能将虚部给抵消，而仅留下实部部分
$$
\cos(2 \pi f_0 t) = \frac{e^{j 2 \pi f_0 t}}{2} + \frac{e^{- j 2 \pi f_0 t}}{2}
$$
同理
$$
\sin(2 \pi f_0 t) = \frac{je^{-j 2 \pi f_0 t}}{2} - \frac{je^{j 2 \pi f_0 t}}{2}
$$
而在复指数函数前面的复系数只是代表着复指数函数旋转的模值和初相。

明白了实信号的复指数表示，对于任意实信号的傅里叶变换都有对称的镜像负频率的意义也就不感到奇怪了，由于傅里叶变换是通过正交基$e^{j2 \pi f t}$来进行分解的，其也就是前面我们所说的基本的复指数信号。对于任意实信号，其的基本成分即复指数信号在复平面上旋转的过程中，总需要一个与自己等大小模、反方向旋转且关于实轴对称的镜像复指数存在，就像前面的余弦信号一样。我们以前所谈论的频率往往只在乎逆时针旋转的正频率，而忽视逆时针旋转的负频率，所以常常会对实信号傅里叶变换而出现负频率而感到十分困惑。基于此，我们只需知道，当我们使用表达式为$F(jw) = \int_{-\infty}^{+\infty}f(t)e^{-jwt}\mathrm{d}t$的傅里叶变换时，我们就已经隐性的将自己的思维置身于复数的世界上了。

我们现在已经对复时域正交信号的概念不再陌生了，其本质就是在复平面上的旋转运动，而实信号作为正交信号的特殊形式，其可视为两组只有旋转方向而其他性质完全相同的复向量的叠加。现在让我们来讨论正交函数在复频域上的形式。先说结论，正交信号在复频域上的形式，就是将组成复时域的所有复指数函数的**初始位置（初始状态）**重新排列在复频域中。举两个例子加深对此的理解，如实偶信号，即时域是实数且满足偶对称的信号，以$\cos(2 \pi f_0 t)$为例，由于其为实信号，故其必有两个关于实轴对称的复指数构成；又由于其为偶信号，所以其的初始状态必在实数轴，因为只有在实轴，才能满足$\pm t$时的幅值相同。再以$\sin(2 \pi f_0 t)$为例，同样由于其为实信号，所以其必然包含关于实轴对称的复指数函数，同时其为奇函数，所以其初相必然在虚轴上（一个向上，另一个向下），所以两个信号的复频域表示如下
![image-20230524170146317](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230524170146317.png)

正交混频（quadrature mixing）过程是使用复指数信号（包含载频信息）和输入信号进行相乘，输入信号不要求其为解析信号或其他，只要是个信号都行。其最终的作用就是导致**输入的频谱左移或右移载频长度**。正交混频的常见应用就是复下混频（complex down-conversion），即对于一个输入的实序列$x(n)$，要将其正交下混频至$x_c(n)$，可以通过让输入信号与复指数函数$e^{-j 2 \pi f_c n t_s}$相乘来实现，其最终的结果必定为正交信号（复信号），所以必须使用双通道的方法来处理，此时实信号的正负频率都会进行相同方向的频移，（若要对复数信号进行正交混频，则需要双输入双输出的复数乘法器），实信号正交混频的框图如下
![image-20230524194355353](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230524194355353.png)

我们使用一个窄带信号来理解正交混频的过程，如下
![IMG_0845(20230524-210712)](D:\qq\820936392\FileRecv\MobileFile\IMG_0845(20230524-210712).JPG)

接下来介绍一个正交采样的例子，该例子用到了上述绝大多数的知识点。正交采样的目的是将一个带载频的窄带信号（其实不窄也行）的窄频信息丢去，即在频域上实现窄带带宽的中心从$f_c$到$0$的转变（此时的信号不再为实信号），并且同时消去讨厌的镜像负频率。正交采样的框图如下
![image-20230524202357143](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230524202357143.png)

仍然使用正交混频的思路，将一个实信号经过正交混频形成一个正交信号，正交信号的频谱即为原来实信号的偶对称频谱左移$f_c$而形成，此时正频率的带宽已经满足正交采样的要求，但是负频率出现了一个很大的高频分量，此时可以使用低通滤波器来滤除，即
$$
x_c(t) = x_c(t) * h_{l}(t) = [i'(t) + jq'(t)] * h_{l}(t) = i'(t) * h_{l}(t) + j[q'(t) * h_{l}(t)]
$$
在模拟信号的处理中，无法对复信号直接进行低通滤波，但根据线性时不变系统的线性性质，是可以分别对正交信号的实部和虚部进行低通滤波，效果等同于最后进行低通滤波。

现实中难以做到I、Q两路平衡，故通常在数字处理中进行正交混频，即对输入信号$x_{bp}(t)$进行采样称为数字信号后，通过数字信号的方法进行正交混频。

> Richard G. Lyons "Understanding Digital Signal Processing"

### 波的各种参量

虽然说这块属于电磁相关的内容，但是挡不住其是在太常见，并且太容易混淆，所以仍然整理在这。

波动具有时间和空间双重周期性，考虑一个波动方程，其表达式为
$$
y = f(x, t) = A\cos(wt - kx + \varphi)
$$
**波长（空间周期）**
时间不变时，上述表达式在空间中的波长，显然$\lambda = \frac{2 \pi}{k}$。
**周期（时间周期）**
空间不变时，上述表达式在时域上的波长，这就回到了最熟悉的时域信号，显然$T = \frac{2 \pi}{w}$。

**波数**
定义为波长（空间周期）的倒数，即$n = \frac{1}{\lambda} = \frac{k}{2 \pi}$，物理意义为单位空间下，波出现的次数。
**频率**
定义为周期（时间周期）的倒数，即$f = \frac{1}{T}$。

**波矢**
波食量来源于波数，波数代表单位空间内波出现的次数，而波矢就是单位空间内波改变的相位量，即$k = 2 \pi n = \frac{2 \pi}{\lambda}$。
**角频率**
时间下的参数，这里不再详述。

**波速**
波速是唯一一个将空间和时间联系在一起的量，其表示波单位时间内，前进的距离。波前进的定义是波的等相位面不断的前进，如上式要保持相位不变，既有$wt - kx + \varphi = \Phi$，即要一直保持相位不变，但是随着时间的增加，$wt$是会不断的增加的，所以此时必须使得$kx$以同速率增加，以保持$wt - kx = 0$不变，此时波速自然产生为$v = \frac{w}{k} = \lambda f$。从表达式中也可以看出，其联系了空间和时间的变化。

### 离散频谱估计

> exact_fft_measurements.pdf

能量重心校准

### DDS

### FM信号的matlab仿真

[两种频率调制(FM)方法的MATLAB实现]: https://www.cnblogs.com/gjblog/p/13494103.html

### 均匀随机变量转任意随机变量

[如何用均匀分布构造正态分布]: https://ziyunge1999.github.io/blog/2020/09/06/constructNormalDistribution/

### 蒙特卡洛性能评估

在《统计信号处理基础》这本书中，需要对估计量的性能做出评估，一种方法便是直接求出该估计量$\hat{\theta}$的PDF，然而解析的方法往往十分复杂甚至无法求出，另一种方法便是使用蒙特卡洛的方法来估计估计量的性能。

对于一个复杂的估计量，例如
$$
\hat{A} = -\frac{1}{2} + \sqrt{ \frac{1}{N} \sum\limits_{n = 0}^{N - 1} x^2[n] + \frac{1}{4} }
$$
其中$x[n] = A + w[n]$，$\mathbf{w} \sim \mathcal{N}(0, \sigma^2 I)$即为高斯白噪声。

我们希望对该随机变量的均值和方差，甚至是PDF做出估计。用计算机模拟的方法可以实现$M$次对于该随机变量的实现，由此，我们通过下式进行估计
$$
\widehat{E[\hat{A}]} = \frac{1}{M} \sum_{i = 1}^{M} \hat{A}_i \\
\widehat{\mathrm{var}(\hat{A})} = \frac{1}{M} \sum_{i = 1}^{M} (\hat{A} - E[\hat{A}])^2
$$
虽然严格意义上来说，方差的估计不是无偏估计，但是为最大似然估计，具有渐近有效的性质。

其次，我们还需要得到该随机变量PDF的估计，我们将一定范围$(x_{\text{min}}, x_{\text{max}})$内的横轴等间隔取$L$个间隔，由此间隔大小$\Delta x = ({x_{\text{max}} - x_{\text{min}}}) / {L}$，得到每一个子区间$(x_i - \Delta / 2, x_i + \Delta / 2)$，对于每一个$\hat{A}$的实现，我们对符合区间的计数加一，若有实现不在整个大的范围之内，那么仍然保持总的计数，而不对任何小区间计数。在matlab中，可以直接使用函数`histogram`进行，详细咨询GPT。

### 拉格朗日数乘法

[拉格朗日乘数法可视化]: https://www.bilibili.com/video/BV1Dd3yeYEzW/

简单来说，在一个有约束优化问题中
$$
\min f(x, y) \\ \text{s.t.} \space q(x, y) \leq z
$$
极值点要么在区域内，要么在边界上，前者与边界无关，直接当作无约束去处理。后者较为复杂：将三维的边界放到二维平面系中，那么边界上的极值点就是二维坐标系曲线导数为0的地方，转换到三维曲线就是方向导数为0的地方。直接求三维曲线的方向导数很复杂，注意到三维曲线方向导数为0点的附近是目标函数某处的等高线（因为等高线的充要是曲线方向导数为0），根据等高线必然垂直与梯度的性质（等高线代表函数值保持不变的方向，梯度代表函数值最大增长的方向，所以等高线和梯度必然垂直），目标函数在该点的梯度与这条曲线的该点垂直。

同时将视角转向约束条件，$q(x, y) = z$可以表示为一个$q(x, y)$上的等高线，同样是等高线垂直于梯度的定理，所以约束函数的等高线垂直于对应梯度。既然目标函数垂直等高线，约束函数也垂直于等高线，而且这两个等高线在极值点附近是一样，所以最后的等价条件为梯度共线
$$
\nabla f(x, y) = \lambda \nabla q(x, y)
$$

