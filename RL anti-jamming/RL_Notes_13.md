## Chapter 13 Policy Gradient Methods

强化学习有关的算法一直分为两种，一是基于价值函数（value based）的算法，通过对最优价值函数进行预测且根据greedy原则来得到最优策略，这也是前面所有章节所讨论的算法。从现在开始我们将开始关注基于策略（policy based）的算法，此时的策略就是参数控制的近似函数$\pi(a\mid s, \boldsymbol{\theta})$。由此，我们需要找到一个标量的损失函数$J(\boldsymbol{\theta})$，用以作为近似函数的优化目标，即
$$
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t+\alpha\widehat{\nabla J(\boldsymbol{\theta}_t)}
$$
其中$\widehat{\nabla J(\boldsymbol{\theta}_t)}$可视为对$\nabla J(\boldsymbol{\theta}_t)$的某种估计量。我们将这种让策略作为近似函数来进行优化的强化学习算法称为策略梯度方法（policy gradient methods）。当策略和价值函数同时使用函数近似方法时，我们称之为actor-critic方法，其中actor和critic分别表示策略和价值函数的近似函数。

### 13.1 Policy Approximation and its Advantage

在动作空间离散且不大的情况下，我们可以使用soft-max的方法来表示策略函数，如下
$$
\pi(a \mid s, \boldsymbol{\theta}) = \frac{\exp[h(s, a, \boldsymbol{\theta})]}{\sum\limits_b\exp[h(s, b, \boldsymbol{\theta})]}
$$
其中$h(s, a, \boldsymbol{\theta})$是对每个状态动作对的一种表征（preference），可以近似的将其视为某个动作的好坏程度，并与价值函数区分开来。参数$\boldsymbol{\theta}$通过控制这个表征来控制着策略函数，这种参数化的方式我们称之为soft-max in action preferences。对于$h(s, a, \boldsymbol{\theta})$来说，我们可以使用前面学到的任何函数近似的模型来表示，包括线性模型以及相应的特征函数、ANN等等。

使用策略梯度这种policy based方法相较于value based方法的一大好处就是前者可以任意的将策略函数表示为任何形式，然而value based下的策略通常只能为$\varepsilon$-greedy方法，这种任意性带来的好处可以由下面这个例子体现。

**Example 13.1**
Short corridor with switched actions，在函数近似的情况下，我们常常遇到的场景就是使用较少的参数来控制更多的状态的策略，比如下图所示的MDP过程

![image-20240227211148954](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240227211148954.png)

我们依次将从左到右的四个小方格称为state 1到state 4。其中state 4为终端状态，其余每个状态都有向左和向右两个情况，正常的状态state 1和state 3的状态转移就是与动作相对应。而state 2的向左动作真正使得状态转移到右边，向右使状态转移到左边。每做一次动作都有奖励-1。假设由于参数过少，三个状态的动作选择都是相同的，即从state 1开始，要么一直向右，要么一直向左。

如果按照$\varepsilon$-greedy的方法，要么向右概率$1 - \varepsilon / 2$很大，要么向左概率很大，其最终导致的state 1的价值都很低。如图所示，只有当向右的概率大概为0.59时，才能使得state 1有最高的初始状态。

同时，不同的环境下对于价值函数或是策略函数近似的难度是不同的，所以要根据环境适当选择使用value based还是policy based。

在实践中，由于$\varepsilon$-greedy基于价值的特性，策略很容易产生剧烈的变化。对于策略梯度方法来说，参数化的函数使得策略在更新的过程中十分平滑。这很大程度上是策略梯度下降拥有更好收敛特性的原因。

### 13.2 The Policy Gradient Theorem

episodic和continuing的任务有着不同的$J(\boldsymbol{\theta})$，但是其最终都可以用一系列共同的公式来描述，我们先来讨论episodic的情况。我们定义episodic任务时的preference measure为初始状态的价值函数，为了表达方便且不失一般性，我们假设初始状态固定为$s_0$，且折扣因子$\gamma = 0$，具体表示如下
$$
J(\boldsymbol{\theta}) = v_{\pi_\boldsymbol{\theta}}(s_0)
$$
通常使用梯度下降来优化这个目标函数，可是该表达式关于参数$\boldsymbol{\theta}$的梯度却难以求出，原因在于：初始状态的价值依赖于之后的轨迹，而该轨迹由取决于动作的选择和状态的转移。动作的选择直接由参数$\boldsymbol{\theta}$控制，同样，状态的转移由于发生在动作选择之后，同样也受参数$\boldsymbol{\theta}$的影响。然而，状态的转移特性于我们而言是未知信息（但是我们可以通过大量地采样间接的直到状态转移特性）。

策略梯度定理（Policy Gradient Theorem）替我们解决了梯度难以求解的问题，其具体形式为以下公式
$$
\nabla J(\boldsymbol{\theta})\propto\sum_s\mu(s)\sum_aq_\pi(s,a)\nabla\pi(a\mid s,\boldsymbol{\theta})
$$
也即目标函数的梯度正比于某个表达式，具体观察这个表达式，若先忽略策略前面的梯度算子，那么该表达式就是**所有状态价值关于状态分布的期望**，而梯度仅仅是增加在策略函数前面。策略梯度定理证明如下
$$
\begin{aligned}
\nabla v_{\pi}(s) &= \nabla\left[\sum_{a}\pi(a\mid s)q_{\pi}(s,a)\right] \\
&= \sum_{a}\Big[\nabla\pi(a\mid s)q_{\pi}(s,a)+\pi(a\mid s)\nabla q_{\pi}(s,a)\Big] \\
&= \sum_{a}\Big[\nabla\pi(a\mid s)q_{\pi}(s,a)+\pi(a\mid s)\nabla\sum_{s',r}p(s',r\mid s,a)\big(r+v_{\pi}(s')\big)\Big] \\
&= \sum_{a}\Big[\nabla\pi(a\mid s)q_{\pi}(s,a)+\pi(a\mid s)\sum_{s'}p(s'\mid s,a)\nabla v_{\pi}(s')\Big] \\
&= \sum_a\Big[\nabla\pi(a\mid s)q_\pi(s,a)+\pi(a\mid s)\sum_{s^{\prime}}p(s'\mid s,a)  \\
& \quad\quad\quad \sum_{a'}\bigl[\nabla\pi(a'|s')q_{\pi}(s',a')+\pi(a'|s')\sum_{s''}p(s''|s',a')\nabla v_{\pi}(s'')\bigr]\Big] \\
&= \sum_{x\in\mathcal{S}}\sum_{k=0}^{\infty}\Pr(s\to x,k,\pi)\sum_{a}\nabla\pi(a\mid x)q_{\pi}(x,a) \\
\end{aligned}
$$
其中$\Pr(s\to x,k,\pi)$代表从状态$s$使用$k$步到达状态$x$的概率，这个式子可以理解成将每层的所有状态价值通过转移概率加权起来，然后最后所有层一起相加。将初始状态$s_0$带入，则得到
$$
\begin{aligned}
\nabla J(\boldsymbol{\theta})& =\nabla v_{\pi}(s_{0})  \\
&=\sum_s\left(\sum_{k=0}^\infty\Pr(s_0\to s,k,\pi)\right)\sum_a\nabla\pi(a\mid s)q_\pi(s,a) \\
&=\sum_s\eta(s)\sum_a\nabla\pi(a\mid s)q_\pi(s,a) \\
&=\sum_{s'}\eta(s')\sum_s\frac{\eta(s)}{\sum_{s'}\eta(s')}\sum_a\nabla\pi(a|s)q_\pi(s,a) \\
&=\sum_{s'}\eta(s')\sum_{s}\mu(s)\sum_{a}\nabla\pi(a\mid s)q_{\pi}(s,a) \\
&\propto\sum_s\mu(s)\sum_a\nabla\pi(a\mid s)q_\pi(s,a)
\end{aligned}
$$
证明完毕。

当任务为continuing时，前面的系数$\sum_{s}\eta(s)=1$（马尔科夫链中应该有相关的公式，有空复习）；对于episodic，系数为每条episode的平均长度。

### 13.3 REINFORCE: Monte Carlo Policy Gradient

前面有提到绝大部分使用带PDF的表达式都可以根据大数定理将表达式样本化，这里将该思想补充得更加完整。若需要将具体表达式转换成样本形式，首先要观察其是否为某种PDF的加权平均和的形式，若有，则可以将其写成期望表达式$\mathbb{E}[\cdot]$，然后将括号内的表达式中的随机变量转换为一次对其的采样（若括号内还有加权平均和的形式，则还有可能写成期望并继续转换为样本形式）。

前面提到，performance measure的梯度$\nabla J(\boldsymbol{\theta})\propto\sum_s\mu(s)\sum_a\nabla\pi(a\mid s, \boldsymbol{\theta}) q_\pi(s,a)$，在梯度下降的表达式中
$$
\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_{t} + \alpha\widehat{\nabla J(\boldsymbol{\theta}_t)}
$$
由于任何系数都可以被包含在$\alpha$中，所以我们可以直接找到关于$\nabla J(\boldsymbol{\theta})$的样本形式。观察该式，可观察其内包含关于$\mu(s)$的加权平均和形式，所以立刻反应其能写出某个随机变量的期望，由于状态$s$是策略$\pi$下的on-policy分布，所以可以写出期望
$$
\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_{\pi}[\sum_a\nabla\pi(a\mid S_t, \boldsymbol{\theta}) q_\pi(S_t, a)]
$$
由此更新表达式可写成（需要同时对策略和价值进行近似）
$$
\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_{t} + \alpha\sum_a \hat{q}(S_t, a, \mathbf{w})\nabla\pi(a\mid S_t, \boldsymbol{\theta}_t)
$$
像这样一次包含了状态$S_t$所有动作的更新表达式，称为all-actions方法。可想而知，其实先起来具有一定困难，在今后的讨论中，这个表达式还会提到。

是否能够将更新表达式再次样本化？观察发现其仍然包含一个加权平均的形式，只不过这次系数不是PDF，而是PDF的梯度。为了符合期望的形式，我们构造相应的PDF，并继续进行推导，如下
$$
\begin{aligned}
\nabla J(\boldsymbol{\theta}) &\propto \mathbb{E}_{\pi}[\sum_a\nabla\pi(a\mid S_t, \boldsymbol{\theta}) q_\pi(S_t, a)] \\
&= \mathbb{E}_{\pi}[\sum_a\pi(a\mid S_t, \boldsymbol{\theta}) q_\pi(S_t, a)\frac{\nabla\pi(a\mid S_t, \boldsymbol{\theta})}{\pi(a\mid S_t, \boldsymbol{\theta})}]  \\
&= \mathbb{E}_{\pi}[q_\pi(S_t, A_t)\frac{\nabla\pi(A_t\mid S_t, \boldsymbol{\theta})}{\pi(A_t\mid S_t, \boldsymbol{\theta})}] \\
&= \mathbb{E}_{\pi}[G_t \frac{\nabla\pi(A_t\mid S_t, \boldsymbol{\theta})}{\pi(A_t\mid S_t, \boldsymbol{\theta})}]
\end{aligned}
$$
故更新表达式可写为
$$
\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_{t} + \alpha G_t \frac{\nabla\pi(A_t\mid S_t, \boldsymbol{\theta})}{\pi(A_t\mid S_t, \boldsymbol{\theta})}
$$
该更新表达式背后有清晰的物理意义，首先$\nabla\pi(A_t\mid S_t, \boldsymbol{\theta})$是策略在$S_t$和$A_t$关于参数$\boldsymbol{\theta}$的梯度，得到的向量代表着在参数空间中朝着$\pi(A_t\mid S_t, \boldsymbol{\theta})$增大最快的方向，也即在状态$S_t$时选择更多$A_t$的方向。如果选择$A_t$更好，也即获得的$G_t$更高，意味着我们需要鼓励选择$A_t$的概率，所以需要$G_t$作为正比的系数。其次，$\pi(A_t\mid S_t, \boldsymbol{\theta})$的概率高，意味着访问$A_t$的频次就更多，从而更多进行更新，即使$A_t$不为最优策略，但多次的更新还是使得选择该动作的概率偏高，从而造成偏差。所以我们需要使用概率$\pi(A_t\mid S_t, \boldsymbol{\theta})$作为反比的系数作为补偿。伪代码如下

![image-20240228114857987](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240228114857987.png)

注意，这里在获取一个episode的数据之后，使用每个单独状态动作对进行一次更新，故使用的是该动作状态对下的单次样本作为对随机变量的估计，所以此时一个episode中对参数进行了多次更新。另一种方式是对一个episode只进行一次更新，此时我们要采用样本的均值作为随机变量的梯度，即
$$
\widehat{\nabla J(\boldsymbol\theta)} = \frac{1}{T} \sum\limits_{t = 0}^{T - 1} G_t \nabla \ln \pi(A_t \mid S_t, \boldsymbol\theta) \\
$$
虽然相比于之前的方式，更新的次数少了，但那一次的更新使用了所有样本数据。同时代码的最后一行，更新的梯度为$\nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta})$​，稍微推导一下，发现其与前面是等价的，但是更简洁。这种向量，我们称之为eligibility vector。

**Exercise 13.3**
计算前面提到了soft-max形式的$\pi(a \mid s, \boldsymbol{\theta})$的梯度，每个$h(s, a, \boldsymbol{\theta})$用线性模型表示$h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^{\top} \mathbf{x}(s, a)$。
$$
\begin{aligned}
\nabla\ln \pi(a \mid s, \boldsymbol{\theta}) &= \nabla\ln \frac{\exp[h(s, a, \boldsymbol{\theta})]}{\sum\limits_b\exp[h(s, b, \boldsymbol{\theta})]} \\
&= \nabla h(s, a, \boldsymbol{\theta}) - \nabla\ln\sum_b\exp[h(s, b, \boldsymbol{\theta})] \\
&= \mathbf{x}(s, a) - \frac{\sum_b\exp[h(s, b, \boldsymbol{\theta})]\mathbf{x}(s, b)}{\sum_b\exp[h(s, b, \boldsymbol{\theta})]} \\
&= \mathbf{x}(s, a) - \sum\limits_b \pi(b \mid s, \boldsymbol{\theta}) \mathbf{x}(s, b)
\end{aligned}
$$

### 13.4 REINFORCE with Baseline

我们可以向前面所叙述的策略梯度定理表达式中添加一基准（baseline）$b(s)$，如下
$$
\nabla J(\boldsymbol{\theta})\propto\sum_s\mu(s)\sum_a\nabla\pi(a\mid s)\big( q_\pi(s,a)-b(s)\big)
$$
若要让该$b(s)$不影响原来的结果，则必须让$b(s)$与$a$无关，原因在于
$$
\sum_a\nabla\pi(a\mid s)b(s) = b(s)\nabla\sum_a \pi(a \mid s) = 0
$$
由此，在经过与前面一样的操作，就能将baseline带入到更新表达式中
$$
\boldsymbol{\theta}_{t + 1} = \boldsymbol{\theta}_{t} + \alpha \big(G_t - b(s)\big) \frac{\nabla\pi(A_t\mid S_t, \boldsymbol{\theta})}{\pi(A_t\mid S_t, \boldsymbol{\theta})}
$$
当对于所有状态都有$b(s)=0$​时，那么该更新表达式于之前的MC策略梯度方法无异。

添加baseline的好处是其能大大降低策略在梯度更新时的方差。例如，假设你遇到了一个状态，其各个动作价值函数都拥有较高的值，那么MC方法得到的$G_t$往往会很高，从而导致策略梯度更新的幅度会比较大。因为该状态普遍拥有高动作价值，所以几乎每次更新的幅度都会很大，这是我们想避免的，所以需要添加一个baseline能调节一次更新的幅度，让这个调节的幅度是相对于当前状态总体情况，而不是绝对情况（优势函数）。同理当状态普遍拥有较低的动作价值时，我们也需要一个baseline将其拉高。总的来说，我们需要的baseline能够拉平不同状态$G_t$差异过大的情况（$G_t$的方差很大），使得每次策略更新的参数增长率尽量拉平，使得尽量免受由于状态所处的前后期所导致的价值差异过大的情况。一个很自然的baseline的选择就是该状态的近似价值函数$\hat{v}(S_t, \mathbf{w})$，即使用$G_t$对策略梯度进行更新的同时还需要对近似价值函数进行更新，最终前面的系数为$G_t - \hat{v}(S_t, \mathbf{w})$，很多地方都称其为优势函数。伪代码如下

![image-20240228153900041](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240228153900041.png)

在这里，我们需要同时调节两次更新的步长$\alpha^{\boldsymbol{\theta}}$和$\alpha^{\mathbf{w}}$，其中关于价值函数近似的梯度更新步长在线性模型下的选择已经由前面章节介绍。对于梯度更新的步长，只能模糊的感觉与$G_t$的方差有关。

![image-20240228154306383](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240228154306383.png)

可以看到，增加了baseline之后的跟新是立竿见影的。并且此时的$\hat{v}(s, \mathbf{w})=w$仅由一个参数决定，换句话说，所有的基准都是相同的。可以预见，更新到最后的$w$就是所有状态价值的均值。

### 13.5 Actor-Critic Methods

前面我们使用了近似函数$\hat{v}(s, \mathbf{w})$作为MC方法的策略梯度更新baseline，那么为什么不将近似状态函数的用途增加，由此我们可以使用基于TD(0)的方法来进行策略梯度更新。将MC方法替换为TD方法的好处在前面叙述了很多，包括bias与variance、样本利用率与样本效率等问题，这里不再赘述。使用TD(0)的策略梯度更新表达式如下
$$
\begin{aligned}
\boldsymbol{\theta}_{t + 1} &= \boldsymbol{\theta}_{t} + \alpha (G_{t : t + 1} - \hat{v}(S_t, \mathbf{w}_t)) \nabla \ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t) \\
&= \boldsymbol{\theta}_{t} + \alpha (R_{t + 1} + \gamma\hat{v}(S_{t + 1}, \mathbf{w}_t) \ - \hat{v}(S_t, \mathbf{w}_t)) \nabla \ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t) \\
&= \boldsymbol{\theta}_{t} + \alpha \delta_t \nabla \ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t) \\
\end{aligned}
$$
由此对应伪代码如下

![image-20240305232131101](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240305232131101.png)

可以看到，仅仅只是将之前REINFORCE with Baseline的算法中MC方法替换为了TD(0)方法。

### 13.7 Policy Parameterization for Continuous Actions

离散动作的情况下，通常将$\pi(a \mid s, \boldsymbol{\theta})$视为一个神经网络，其输入某个状态$s$，输出一个离散分布列，代表各个动作选择的概率。但是当动作空间巨大时，输出的离散分布列也随之很大。解决这一困难的方法就是将输出视为一个pdf，这个pdf由网络的参数所控制。
