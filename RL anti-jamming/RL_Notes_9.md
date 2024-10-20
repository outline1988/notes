## Chapter 9 On-policy Prediction with Approximation

前面关于强化学习的算法都基于tabular，即我们可以将状态列成表格，使用每一步的数据更新表格中对应状态的价值，其他状态的价值保持不变。然而实际问题的状态数有时十分庞大，表格的方法面临计算量和存储量的巨大困难，无法继续使用。本章中我们重点考虑prediction问题，即给定$\pi$，我们希望求出所有状态的价值函数（或近似价值函数）。我们开始考虑function approximation的方法来对价值函数进行近似。即用近似函数$\hat{v}(s, \mathbf{w})$去近似真实的该状态的价值函数$v_{\pi}(s)$
$$
\hat{v}(s, \mathbf{w})\approx v_{\pi}(s)
$$
我们的目标是找到一个合适的$\mathbf{w}$使得近似函数$\hat{v}(s, \mathbf{w})$能够最接近真实的价值函数$v_{\pi}(s)$。通常情况下，若设权重向量$\mathbf{w}$的长度为$d$（即参数的数量），其要远远小于状态的数量$|\mathcal{S}|$。所以在每一次以改变参数$\mathbf{w}$为目的的更新中，不同于表格型解决方法一次更新只影响一个状态的价值的情况，函数近似的一次更新一定会导致多个状态价值的同时更新。我们希望能用少量的数据（训练集不能覆盖所有的状态），使得近似函数$\hat{v}(s, \mathbf{w})$有一种泛化（generalize）能力，即使是训练集中没有出现的状态，近似函数也能尽可能精确的估计该状态的价值。

### 9.1 Value-function Approximation & 9.2 The Prediction Objective 

我们首先引入一个记号$s \mapsto u$（自变量到因变量的逼近值），其表示我们要让状态$s$的价值函数更加趋向于$u$，例如每一次的MC更新，我们有$S_t \mapsto G_t$；每一次TD的更新，我们有$S_t \mapsto R_{t + 1} + \gamma V(S_{t + 1})$。我们将这样的一种行为描述为一种input-output的对应关系。有监督学习（supervised learning）就是想要去拟合这种输入和输出的关系。如果当$u$为一个数时，有监督学习的过程就被称为函数近似（function approximation）。

我们可以将有监督学习的函数近似问题套用到强化学习中的价值函数近似问题上。我们将强化学习中每一次更新的数据作为数据集，并使用有监督学习的方法进行函数近似，就可以得到价值函数的近似。**将有监督学习的算法使用在强化学习上，至少需要克服三个问题，一是与一般回归问题不同，强化学习所用到的数据集$s \mapsto u$的目标函数$u$不是真正的价值函数$v_{\pi}(s)$，而是价值函数的某个估计，而且该估计量通常方差很大，例如$R_{t + 1} + \gamma V(S_{t + 1})$和$G_t$，但是一般的有回归问题的输入输出的输出也是真值增加一个噪声，所以有监督学习的方法从天然就能够克服这个问题；二是由于强化学习在线（online）更新的特性，所以需要用增量式（序贯）的方式进行学习，增量式的问题可以很巧合的通过随机梯度下降来克服；三是要克服非平稳的问题（环境MDP在不断的发生改变），训练集会随着策略的更新，$s \mapsto u$中的$u$会不断改变，与以往的静态有监督学习算法不同，目前我们不考虑MDP会发生改变的情况。**

在prediction问题中，为了评估$v_\pi(s)$和$\hat{v}(s,\mathbf{w})$近似的程度，我们使用以下损失函数
$$
\overline{\mathrm{VE}}(\mathbf{w})\doteq\sum_{s\in\mathcal{S}}\mu(s)\left[v_\pi(s)-\hat{v}(s,\mathbf{w})\right]^2
$$
其中$\mu(s)$从某种程度上代表着每个状态的重要程度（通常用MDP的访问频次来定义），我们称之为on-policy distribution；而后面的$\left[v_\pi(s)-\hat{v}(s,\mathbf{w})\right]^2$代表着每一个状态的近似函数与真值之间的距离。所以$\overline{\mathrm{VE}}$可用来粗略的代表价值近似函数与真值之间的近似程度。上述损失函数与一般的有监督学习的问题的平方损失函数是完全相同的，后者在表达式中没有出现$\mu(s)$，是因为后者直接使用训练集中所有的数据来求出误差，而训练集中的数据出现的频次在此问题中为$\mu(s)$。

对于给定的训练数据集，我们要找到一个最合适的$\mathbf{w}^*$，使得对于所有的$\mathbf{w}$，都有$\overline{\mathrm{VE}}(\mathbf{w}) \geq \overline{\mathrm{VE}}(\mathbf{w}^*)$，即全局最优解。然而实际的关于$\mathbf{w}$的复杂形式往往使得这一问题几乎不可能求解，所以我们退而求其次，允许局部最优解的出现。

目前上没有很大的定论关于让$\mathrm{VE}$最小是使得policy improvement最好的选择，不过目前就使用这个损失函数。

### 9.3 Stochastic-gradient and Semi-gradient Methods

我们可以使用梯度下降法（Gradient Descent）来找到合适的$\mathbf{w}$，理论上我们只要进行以下迭代式即可
$$
\mathbf{w}_{t + 1} = \mathbf{w}_t - \frac{1}{2} \alpha \nabla \overline{\mathrm{VE}}(\mathbf{w})
$$
本质上$\overline{\mathrm{VE}}(\mathbf{w})$是求期望的表达式，随机变量为$s$，其PDF就是不同状态出现的频次$\mu(s)$，所以按照随机梯度下降（Stochastic Gradient Descent）的理论，我们可以使用一次的样本值$S_t$和$v_{\pi}(S_t)$来进行更新，注意到第一行能写到第二行是因为$\mathbf{w}_t$与$v_{\pi}(S_t)$是不相关的。
$$
\begin{aligned}
\mathbf{w}_{t + 1} &= \mathbf{w}_t - \frac{1}{2} \alpha \nabla \left[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\right]^2 \\
&= \mathbf{w}_t + \alpha \left[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\right] \nabla \hat{v}(S_t, \mathbf{w}_t)
\end{aligned}
$$
这里书上提供了一个SGD的理解方式，SGD每次使用一小批样本进行梯度下降，若只使用这一小批样本进行梯度下降，那么结果就是能使得近似函数能够在这小批数据的误差最小，也即参数空间中存在一点关于本样本的局部最优点，使用这个特定样本进行梯度下降更新的结果就是这个局部最优点。但是我们的样本是大量的，将大量的样本分成多个batch（也就是SGD进行更新的batch），那么每个batch在参数空间中都存在一个局部最优点。这些点通常不会相同，离散的驻足在参数空间中，并且每个局部最优点都与起点有一条下降路径的连线。使用所有样本的梯度下降在参数空间的某个点进行更新时，前进的方向是所有batch方向的均值，所以所有样本进行梯度下降所达到的局部最优点是前面所有batch局部最优点的balance，也就是均值。SGD的想法就是每次只前进一个batch的方向，在所有的batch都前进之后，自然而然地回达到那个balance的局部最优点。对于梯度下降的学习率$\alpha$来说，我觉得其在绝大多数的情况下只影响表达式收敛时的精度，而不会影响前进的大方向，所以我认为，若已经有了参数空间上起点和终点大致距离的先验信息，那么很容易想到在（随机）梯度下降的过程中，应该先使用较大的学习率，等到快到终点时，在减小学习率，由此既能达到更快的收敛速度，又能达到更高的收敛精度。

然而现实中，我们难以获得状态的真实价值$v_\pi(S_t)$，理论告诉我们可以用一个状态价值的**无偏估计量$U_t$**来进行替代，如MC方法中的$U_t = G_{t}$
$$
\mathbf{w}_{t + 1} = \mathbf{w}_t + \alpha \left[U_t - \hat{v}(S_t, \mathbf{w}_t)\right] \nabla \hat{v}(S_t, \mathbf{w}_t) \\
U_t = v_{\pi}(S_t) + n
$$
其中$n$为噪声。

我们无法用TD方法得到的估计量来代替$v_\pi(S_t)$，原因是$R_{t + 1} + \gamma \hat{v}(S_{t + 1}, \mathbf{w}_t)$是有偏估计量（因为对其求期望是对随机变量$R_t$求期望，而此时$\hat{v}(S_{t + 1}, \mathbf{w}_t)$被视为不含有随机性且不等于真值$v_{\pi}(S_{t + 1})$的常量）；其次，$\hat{v}(S_{t + 1}, \mathbf{w}_t)$包含了$\mathbf{w}_t$，故求导后$\mathbf{w}_t$不能与$R_{t + 1} + \gamma \hat{v}(S_{t + 1}, \mathbf{w}_t)$分离。

但是如果我们不考虑$\hat{v}(S_{t + 1}, \mathbf{w}_t)$有偏且包含$\mathbf{w}_t$的问题，可写出半随机梯度下降（Semi-Gradient Descent）的表达式
$$
\mathbf{w}_{t + 1} = \mathbf{w}_t + \alpha \left[R_{t + 1} + \gamma \hat{v}(S_{t + 1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)\right] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$
在这里我们并没有求出真正的梯度值，所以这不能算是正规的梯度下降方法（故我们称他为半梯度下降方法），该方法只考虑了改变$\mathbf{w}_t$会对价值近似函数的影响，而忽略了$\mathbf{w}_t$对目标值得影响。

**Example 9.1**
我们考察一个具体的一千个状态的随机游走问题，有一个1000长的方格，在其中的一格中，你有等概率向左边一百和右边一百的格子跳跃，若左边或右边没有一百个格子，那么最边缘的格子会积累那些消失的格子的概率。若使用state aggregation和MC方法估计价值函数。其中从左到右的格子设置为state 1-1000，并且设置每100个状态为一类，即state 1-100为一类，state 101-200为另一类，以此类推。

由于使用了state aggregation的模型，所以$\hat{v}(s, \mathbf{w})$中的$\mathbf{w}$是个长度为10的向量，其中每一个参数代表某一类的价值函数预测值，比如第一个元素$w_1$代表着state 1-100所共享的值。因此$\nabla \hat{v}(s, \mathbf{w})$也是一个长度为10的向量，并且其十个元素中有九个元素为0，只有在状态$s$所处的类别处对应的$\mathbf{w}$元素为1，由此根据MC方法的随机梯度下降的表达式
$$
\mathbf{w}_{t + 1} = \mathbf{w}_t + \alpha [G_t - \hat{v}(S_{t}, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$
在$\nabla \hat{v}(S_t, \mathbf{w}_t)$为0的行中，$w_{t + 1} = w_t$保持不变；而在$\nabla \hat{v}(S_t, \mathbf{w}_t)$为1的行中，有如下式子
$$
w_{t + 1} = w_t + \alpha [G_t - \hat{v}(S_{t}, \mathbf{w}_t)]
$$
即对$S_t$所处类的价值函数进行了普通的更新。在使用了大量样本进行训练之后，有以下的结果

![image-20240117095508973](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240117095508973.png)

可以先使用其他方法（解析法）来对随机游走的真值$v_{\pi}(s)$进行求解，然后在使用MC方法结合状态聚类的方式进行对价值函数的prediction。prediction问题的最终结果是使得真值和预测值的均方误差最小，有图所见，阶梯状的蓝色曲线确实尽可能地贴合红色的真实价值函数曲线。

注意，SGD取得的结果就是使得所有样本的均方误差达到最小，不过由于样本中包含的不同状态的频次不同，如本例中state 500有着最大的频次而最边缘的state拥有最少的频次，才会使得最终的结果是$\overline{\mathrm{VE}}(\mathbf{w})\doteq\sum_{s\in\mathcal{S}}\mu(s)\left[v_\pi(s)-\hat{v}(s,\mathbf{w})\right]^2$最小，这个表达式就等效于所有样本的均方最小。结论是：就按照随机梯度下降那一套进行训练，最终整个样本的均方误差最小等效于$\overline{\mathrm{VE}}$的最小。

### 9.4 Linear Methods

正如有监督学习中的回归问题的一样，关于$v_{\pi}(s, \mathbf{w})$的第一个模型就是线性模型，其为关于权重的线性函数，即
$$
v_{\pi}(s, \mathbf{w}) = \mathbf{w}^{\top} \mathbf{x}(s)
$$
其中$\mathbf{x}(s) = [x_1(s), x_2(s), \cdots, x_d(s)]^\top$为一个列向量，其内每一个元素都是一个从状态到实数轴的映射，我们称函数$x_i(\cdot)$为基函数（或是feature）。由于线性模型$v_{\pi}(s, \mathbf{w})$的线性性质，根据SGD的更新表达式
$$
\begin{align}
\mathbf{w}_{t + 1} &= \mathbf{w}_t + \alpha [U_t - \mathbf{w}^\top \mathbf{x}(S_t)] \nabla \mathbf{w}^\top \mathbf{x}(S_t) \\
&= \mathbf{w}_t + \alpha [U_t - \mathbf{w}^\top \mathbf{x}(S_t)] \mathbf{x}(S_t) \\
\end{align}
$$
对于线性模型，$v_{\pi}(s, \mathbf{w})$关于$\mathbf{w}$只有一个全局最优点（global optimum），那么任何能够使得$\mathbf{w}$收敛到局部最优点（local optimum）的的方法（例如SGD）都能使得线性模型收敛至全局最优点。更特别的，前面提到的semi-gradient方法由于不是真正的梯度下降方法，所以常常无法收敛，但是在线性模型的假设下，semi-gradient TD(0) 方法能够收敛到一个TD不动点（TD fixed point）$\mathbf{w}_{\mathrm{TD}}$，这个不动点在局部最优点的附近，计算的方式如下
$$
\begin{aligned}
\mathbf{w}_{t+1}& \doteq\mathbf{w}_{t}+\alpha\Big(R_{t+1}+\gamma\mathbf{w}_{t}^{\top}\mathbf{x}_{t+1}-\mathbf{w}_{t}^{\top}\mathbf{x}_{t}\Big)\mathbf{x}_{t}  \\
&=\mathbf{w}_{t}+\alpha\Big(R_{t+1}\mathbf{x}_{t}-\mathbf{x}_{t}\big(\mathbf{x}_{t}-\gamma\mathbf{x}_{t+1}\big)^{\top}\mathbf{w}_{t}\Big),
\end{aligned}
$$
为了简化，$\mathbf{x}(S_t) = \mathbf{x}_t$。由于对于MDP过程来说，时刻$t$的状态$S_t$是一个随机变量，所以$\mathbf{x}_t$也是随机变量，故对于更新表达式，$\mathbf{w}_t$同样是随机变量，每次的更新都是随机变量的某个取值，只不过随着时间和样本数的增加，$\mathbf{w}_t$的方差越来越小，直到收敛。故求其不动点应该是在期望的意义下进行的，如下
$$
\begin{align}
\mathbb{E}[\mathbf{w}_{t +1} \mid \mathbf{w}_{t}] &= \mathbf{w}_t + \alpha \big(\mathbb{E}[R_{t + 1}\mathbf{x}_t] - \mathbb{E}[\mathbf{x}_t(\mathbf{x}_t - \gamma \mathbf{x}_{t + 1})^\top] \mathbf{w}_t \big) \\ 
&= \mathbf{w}_t+\alpha(\mathbf{b}-\mathbf{A}\mathbf{w}_t)
\end{align}
$$
由此可求其在期望意义下的不动点
$$
\begin{aligned}
\mathbf{b}-\mathbf{A}\mathbf{w}_\mathrm{TD}&=\mathbf{0}\\
\mathbf{b}&=\mathbf{A}\mathbf{w}_\mathrm{TD}\\
\mathbf{w}_\mathrm{TD}&\doteq\mathbf{A}^{-1}\mathbf{b}
\end{aligned}
$$
数学可以证明，线性模型的TD fixed point一定存在，并且semi-gradient TD(0)在取得合适的$\alpha$的情况下（$\alpha$不断减小至0）一定能收敛，具体证明要使用正定性与特征值的关系，见书P206页。

数学还可以证明，TD semi-gradient算法收敛到的TD不动点$\mathbf{w}_{\mathrm{TD}}$所能达到的均方误差$\overline{\mathrm{VE}}(\mathbf{w}_{\mathrm{TD}})$虽然不是最小，但是其拥有一上界，这个上界是和全局最优点的误差是成一定比例的
$$
\overline{\mathrm{VE}}(\mathbf{w_\mathrm{TD}}) \leq \frac{1}{1-\gamma}\min_\mathbf{w}\overline{\mathrm{VE}}(\mathbf{w})
$$
通常来说，在continuing tasks中，$\gamma$接近于1，所以$\overline{\mathrm{VE}}(\mathbf{w_\mathrm{TD}})$于最小误差仍然拥有很大的距离。由前面的章节可知，TD方法大量提高了收敛的速度，并且使用该方法的价值函数的预测值拥有最大似然估计量的性质，但是在此处由于semi-gradient并不是正真梯度下降的限制，又导致了其精度上的损失。

**Example 9.2**
我们讲semi-gradient TD算法运用在Example 9.1的随机游走问题上。Example 9.1所使用的state aggregation模型本质上是线性模型的特例，即$v_{\pi}(s, \mathbf{w}) = \mathbf{w}^{\top} \mathbf{x}(s)$，每一个权值代表着每一个类的代表值，而feature向量$\mathbf{x}(s)$将同属一个类别的状态映射为1，不为一个类别的的状态映射为0。使用TD semi-gradient方法并且状态分类情况同Example 9.1，得到的结果如下

![image-20240118115927553](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240118115927553.png)

可以很清楚看到，semi-gradient TD没有收敛到全局最优点，但也与真实值相差不大，好处是大大提高了收敛速度。

右图将使用state aggregation的1000状态随机游走问题与使用n-step TD方法的19状态随机游走问题进行了匹配。为了更好的匹配，需要将state aggregation的类别增加至20，由此每一类都只有50个状态。由于1000状态的随机游走问题每次向左或向右等概率抽取100之内的状态进行转移，从期望意义下，状态向左或向右转移50格，这与19状态的随机游走问题每次向左或向右走一步刚好相同，所以可以看到结果图也是十分类似的。从图中可以看出，当$n=4$时，semi-gradient TD方法能在10个episode的情况下就将价值函数的估计值控制在0.3以下。

使用n-step TD方法的半梯度下降更新方程如下
$$
\mathbf{w}_{t+n}\doteq\mathbf{w}_{t+n-1}+\alpha\left[G_{t:t+n}-\hat{v}(S_t,\mathbf{w}_{t+n-1})\right]\nabla\hat{v}(S_t,\mathbf{w}_{t+n-1})
$$
其中
$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}\hat{v}(S_{t+n},\mathbf{w}_{t+n-1})
$$

### 9.5 Feature Construction for Linear Methods

前面提到，我们将$\mathbf{x}(s)$称之为基函数，其中$s$抽象地表征状态，而$\mathbf{x}(\cdot)$将这一状态转化为一个特征向量，故我们也称之为特征函数。注意每个特征并不一定就是表示着多维状态空间中每一维的信息，例如二维状态空间中的状态$s$，其包含两个状态信息$s_1$和$s_2$，而我们可以将这个状态映射为$\mathbf{x}(s) = [1, s_1, s_2, s_1^2, s_2^2, s_1 s_2]^{\top}$。选择特征是将先验知识添加进RL问题中的方式之一。

就单独对线性模型来说，其一个坏处就是难以将不同特征之间的交互考虑在内，因为每一个特征都只与其相应的权重有关，所以若是特征函数没有将交互的状态信息设置为一个特征，那么线性模型就再无法考虑特征之间的交互了。

**Polynomials**
若状态$s$包含$k$个信息量$s_1, s_2, \cdots, s_k$，即$k$维状态空间，则$n$阶polynomials基函数特征$x_i$表示如下
$$
x_i(s) = \prod_{j=1}^k s_j^{c_i, j}
$$
其中$c_{i, j}$取自集合$\{0, 1, \dots, n\}$，对于状态的单个信息，其最高阶就是$n$，故每个信息量$s_i$有$n + 1$种选法，共有$k$个信息量，则共有$(n + 1)^k$个特征。

例如$n = 2$且$k = 2$，则$\mathbf{x}(s) = [1, s_2, s_2^2, s_1, s_1 s_2, s_1 s_2^2, s_1^2, s_1^2s_2, s_1^2s_2^2]^{\top}$共9个特征。

**Fourier Basis**
傅里叶级数即将某一周期函数表示成正弦函数和余弦函数的加权和。同样，对于一段有限长的非周期函数，可以首先按照该函数的长度进行周期延拓后再进行周期函数的近似。若将有限长的非周期函数以该函数长度的两倍进行周期延拓，前半段用函数本身，后半段用函数的镜像，可将该函数周期延拓为偶函数，从而仅使用余弦函数进行加权和。

以一维函数的近似为例，该函数非周期且限制在$[0, 1]$。则使用以下的基函数
$$
x_i(s) = \cos(i\pi s), ~~ s \in [0, 1]
$$
其中$i=0, 1 \dots, n$，而$n$表示傅里叶近似的阶数。

将每个状态$s$所包含的信息量写成向量的形式，即$\mathbf{s} = [s_1, s_2, \cdots, s_k]^{\top}$，且$s_i \in [0, 1]$则多维傅里叶基函数的表达式如下
$$
x_i(s) = \cos(\pi \mathbf{s}^{\top}\mathbf{c}^i)
$$
其中$\mathbf{c}^i = [c_1^i, c_2^i, \cdots, c_k^i]$也为长度为$k$的向量，其内的每一个元素$c_j^i \in \{0, \cdots , n\}$具有$n+ 1$个取值，表示沿着当前信息量轴的频率成分。故$k$维的$n$阶傅里叶基函数有数量$(n + 1)^k$。综上，傅里叶基函数的特征为对于每个状态的信息量拥有特定频率的余弦函数。

例如$n = 2$且$k = 2$，则$\mathbf{x}(s) = \cos(\pi [1, s_2, 2s_2, s_1, s_1 + s_2, s_1 + 2s_2, 2s_1, 2s_1 + s_2, 2s_1 + s_2]^{\top})$​共9个特征。

当使用傅里叶基函数进行随机梯度下降的更新时
$$
\mathbf{w}_{t + 1} = \mathbf{w}_t + \alpha [U_t - \mathbf{w}_t^{\top} \mathbf{x}(s)] \mathbf{x}(s)
$$
梯度下降的更新步长建议选择为$\alpha_i = \alpha/\sqrt{(c_1^i)^2+\ldots+(c_k^i)^2}$，此时每个权值都拥有不同的更新步长。

在对于连续函数的近似上，傅里叶基函数都具有良好的表现，但是在不连续的函数上，由于需要更高频的基函数，导致此时近似的精度下降。

与多项式基类似，傅里叶基的数量也呈指数增长，当状态信息量少时，可以使用全部的傅里叶基；当状态信息量多时，就需要使用先验知识来选择合适的基函数子集，这同样也是添加先验知识的方式。同时选择傅里叶基的另外一个好处是，可以很方便的通过调控$\mathbf{c}^i$来把握当前傅里叶基所表示的特征（比如噪声）或是状态之间的交互程度。但是傅里叶基的坏处就是其都是全局函数，从而难以代表局部的特征（适用于特征更加平稳的函数）。

下图为在随机游走问题上使用傅里叶基函数的例子，可以看出使用傅里叶基拥有更高的精度。

![image-20240220210032239](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240220210032239.png)

**Coarse Coding & Tile Coding**
回想我们在1000随机游走问题上使用的状态聚类方法，由于状态过多，我们将位置相近的状态视为同一类，这一类状态具有相同的一个参数，而这个参数最终控制了该类的价值函数（参数值等于该类价值函数），我们通过MC方法来对每个类别的参数进行迭代逼近，也就是对每个类的价值函数进行预测。现在想象一个二维的状态空间，二维空间中的每一个点都存在一个价值函数，需要由我们来进行估计，我们同样可以使用状态聚类的方法将整个二维空间分成若干个小方格。现在考虑在这二维空间中以如下方式添加圆

![image-20240223163858667](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223163858667.png)

我们使用可重叠的圆在平面中划分出了不同的区域，每个小区间都是若干个圆的一小部分。我们令每一个圆都对应一个参数，并且让每一个小区间的价值函数为当前区间所在若干圆对应的若干参数之和。等效于让$\mathbf{x}(s)$为一个将状态映射为与圆数量长向量的函数，向量的每个元素值特定方式给出：给定一个$s$，判断这个$s$被哪些具体的圆所包含，并相应让这若干圆对应的向量元素为1，其余为0。

模型在学习的时候，给定一个状态和对应的价值，则该更新只对所在圆对应的参数有影响，而这些圆所对应的参数又不光只影响一个区间，而影响了这若干圆包括的所有范围。例如上图中的$s$进行了更新，则会有3个圆对应的参数发生变化，由此造成了这三个圆所覆盖的地方都发生了变化（灰色部分）。对于$s'$来说，包含$s'$的两个圆有一个发生了变化，所以相当于这个区域也进行了更新，但并没有$s$的更新强度那么高，因为$s'$​的另一个圆没有收到影响。总体上来说，更新的范围由中心向外扩张，强度则逐渐减弱，并且我们可以通过调控圆的不同位置来改变泛化的方式。

圆越大，所能影响的范围就越广，也即泛化能力越强。直觉上，若泛化能力强，会造成部分区域在细节上有所欠缺，但是这样的担心是没有必要的，泛化能力的强弱对于收敛时的精度影响不大，但对前期的速度影响很大。

**Example 9.3**
我们使用不同大小的区间来对一个们函数进行拟合，如下图

![image-20240223171344002](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223171344002.png)

学习率选择为$\alpha = 2 / n$，这里的$n$​​表示每个小区间被包含的数量，之所以要这样选择，这是为了左中右三组试验拥有相同的学习率，若一个小区间被包含的数量过多，那么一次更新就同时更新同等数量的参数，而各个参数之间的关系都是简单的加和，所以需要除上数量使得该小区间一次的学习率相同。实验展现的结论就是前面所述。

若将上述的圆换成一个个等大的小方格，这就是tile coding

![image-20240223172203072](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223172203072.png)

具体的影响方式如下

![image-20240223172259282](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223172259282.png)

这里只写了大概，需要更加详细的了解看书P217。
